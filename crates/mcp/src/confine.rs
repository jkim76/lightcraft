//! File access confinement for the headless MCP server (`lightcraft-cli mcp
//! --automation-read-root DIR --automation-write-root DIR`, as in PhotoCraft, EffectCraft and
//! FilmCraft): an agent may only read files under a read root (or a write root) and only write
//! under a write root.
//!
//! Every command an agent runs through the headless backend ([`crate::Headless`]: `command_run`,
//! each step of `command_batch`, the helper tools and the per-command `cmd_*` tools) is checked
//! before it runs:
//!
//! - the file parameters it reads ([`READS`]) and writes ([`WRITES`]) must be absolute paths
//!   inside the roots: no `..`, symbolic links resolved through the deepest existing ancestor;
//! - any other string parameter that is an absolute path is checked as a read;
//! - commands that write the photos' own files or files next to them (sidecars, `-copy`,
//!   DNG conversion, merges, renames, moves: [`WRITES_ORIGINALS`]) need those photos' originals
//!   inside a write root, as does `library.import` with `mode: "move"` for its sources;
//! - an export needs a destination (`path` or `dir`, else it would write to the server's working
//!   folder) inside a write root, or with `destinationMode: "sameAsOriginal"` originals inside one;
//! - opening another application ([`REFUSED`]) and turning on automatic XMP sidecars (which
//!   writes next to every edited original) are refused.
//!
//! In a confined session the photo decoders read only inside the roots too
//! ([`Confinement::confine_loaders`]), so a library opened with `--library` that references
//! originals elsewhere renders those as unavailable.
//!
//! Known limits: the app's own data folder (presets, templates, models, the lens database) is
//! not a user file and stays writable; files a library already references are still read by
//! the commands that read metadata or copy originals (`format: "original"` exports).

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use lightcraft_engine::Session;
use lightcraft_engine::catalog::{PhotoId, Source};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// Commands refused while confined: they open another application.
pub const REFUSED: &[&str] = &["photo.editExternal"];

/// Commands that read files or folders named by these parameters.
pub const READS: &[(&str, &[&str])] = &[
    ("library.import", &["paths"]),
    ("library.importPreview", &["paths"]),
    ("library.inspectLightroom", &["path"]),
    ("library.importLightroom", &["path"]),
    ("library.browse", &["path"]),
    ("library.autoImport", &["folder"]),
    ("library.findMissing", &["folder"]),
    ("folder.scanChanges", &["path"]),
    ("folder.synchronize", &["path"]),
    ("photo.relink", &["path"]),
    ("photo.autoTagTracklog", &["path"]),
    ("curve.importPresets", &["path", "paths"]),
    ("mask.importEffects", &["path", "paths"]),
    ("preset.import", &["paths"]),
    ("profile.import", &["paths"]),
    ("keyword.import", &["path"]),
    ("print.template.import", &["path"]),
    ("slideshow.templates.import", &["path"]),
    ("denoise.models.inspect", &["path"]),
    ("denoise.models.install", &["path"]),
    ("faces.models.inspect", &["path"]),
    ("faces.models.install", &["path"]),
    ("metadata.placeNamesStatus", &["dataset"]),
    ("metadata.reverseGeocode", &["dataset"]),
    ("export.addToPhotos", &["paths"]),
];

/// Commands that write files or folders named by these parameters.
pub const WRITES: &[(&str, &[&str])] = &[
    ("app.export", &["path", "dir"]),
    ("library.import", &["destination"]),
    ("library.smartPreviewsLocation", &["path"]),
    ("curve.exportPresets", &["path"]),
    ("mask.exportEffects", &["path"]),
    ("preset.export", &["path"]),
    ("keyword.export", &["path"]),
    ("print.exportPdf", &["path"]),
    ("print.template.export", &["path"]),
    ("export.contactSheet", &["path"]),
    ("export.checkTarget", &["path"]),
    ("slideshow.export", &["path"]),
    ("slideshow.templates.export", &["path"]),
    ("slideshow.preview", &["path"]),
    ("merge.hdr", &["previewPath"]),
    ("merge.panorama", &["previewPath"]),
    ("merge.hdrPanorama", &["previewPath"]),
    ("folder.create", &["parent"]),
    ("folder.move", &["path", "into"]),
    ("folder.rename", &["path"]),
    ("photo.moveToFolder", &["folder"]),
];

/// Commands that write the photos' original files or files next to them (`ids`, else the
/// selection): those originals must be inside a write root. `library.retrySidecars` writes the
/// sidecars of any photo, so every photo of the library counts.
pub const WRITES_ORIGINALS: &[&str] = &[
    "photo.duplicate",
    "photo.convertToDng",
    "photo.saveMetadataToFile",
    "photo.rename",
    "photo.moveToFolder",
    "merge.hdr",
    "merge.panorama",
    "merge.hdrPanorama",
    "merge.hdrBatch",
    "library.retrySidecars",
];

/// Parameters that hold text, names or templates, never file paths: not checked as paths even
/// when they start with `/`.
const NOT_PATHS: &[&str] = &[
    "text", "name", "title", "caption", "label", "keyword", "keywords", "template", "naming", "rename", "organize", "query", "filter", "to", "from",
];

/// The folders an agent may read and write.
#[derive(Clone, Debug, Default)]
pub struct Confinement {
    read: Vec<PathBuf>,
    write: Vec<PathBuf>,
}

impl Confinement {
    /// Roots must be existing folders; they are stored resolved (symbolic links followed).
    pub fn new(read: &[PathBuf], write: &[PathBuf]) -> Result<Self, String> {
        let canon = |p: &PathBuf| -> Result<PathBuf, String> {
            let c = p.canonicalize().map_err(|e| format!("automation root {}: {e}", p.display()))?;
            if !c.is_dir() {
                return Err(format!("automation root {} is not a folder", p.display()));
            }
            Ok(c)
        };
        Ok(Self { read: read.iter().map(canon).collect::<Result<_, _>>()?, write: write.iter().map(canon).collect::<Result<_, _>>()? })
    }

    pub fn read_roots(&self) -> &[PathBuf] {
        &self.read
    }

    pub fn write_roots(&self) -> &[PathBuf] {
        &self.write
    }

    /// Whether `path` may be read or written: absolute, without `..`, and inside a root once
    /// symbolic links in its existing part are resolved. Reads may use the write roots too.
    pub fn allows(&self, path: &Path, access: Access) -> bool {
        let Some(resolved) = resolve(path) else { return false };
        let mut roots = match access {
            Access::Read => self.read.iter().chain(self.write.iter()),
            Access::Write => self.write.iter().chain([].iter()),
        };
        roots.any(|r| resolved.starts_with(r))
    }

    /// `Ok` when `path` may be accessed, else the error an agent sees (naming the parameter).
    pub fn check_path(&self, what: &str, path: &str, access: Access) -> Result<(), String> {
        if self.allows(Path::new(path), access) {
            return Ok(());
        }
        let (verb, roots) = match access {
            Access::Read => ("read", "read or write roots"),
            Access::Write => ("write", "write roots"),
        };
        Err(format!("{what} {path} is outside the {roots}; file access is confined (it may only {verb} absolute paths inside them)"))
    }

    /// Refuse a command that would reach outside the roots. `id` and `params` as given to the
    /// engine; `session` answers which photos a command acts on.
    pub fn check_command(&self, session: &Session, id: &str, params: &Value) -> Result<(), String> {
        if REFUSED.contains(&id) {
            return Err(format!("`{id}` is not available while file access is confined (--automation-read-root/--automation-write-root)"));
        }
        let turns_on_sidecars = match id {
            "library.toggleAutoWriteXmp" => !session.xmp.auto_write,
            "library.xmpPreferences" => params.get("autoWrite").and_then(Value::as_bool) == Some(true),
            _ => false,
        };
        if turns_on_sidecars {
            return Err(format!(
                "`{id}`: automatic XMP sidecars can't be turned on while file access is confined (they are written next to every edited original)"
            ));
        }
        let reads = keys_for(READS, id);
        let writes = keys_for(WRITES, id);
        if let Some(obj) = params.as_object() {
            for (k, v) in obj {
                let access = if writes.contains(&k.as_str()) {
                    Some(Access::Write)
                } else if reads.contains(&k.as_str()) {
                    Some(Access::Read)
                } else {
                    None
                };
                match access {
                    Some(a) => self.check_value(id, k, v, a, true)?,
                    None => self.check_value(id, k, v, Access::Read, false)?,
                }
            }
        }
        if id == "library.import" && params.get("mode").and_then(Value::as_str) == Some("move") {
            // a move deletes the sources once they are copied
            self.check_value(id, "paths", params.get("paths").unwrap_or(&Value::Null), Access::Write, true)?;
        }
        if WRITES_ORIGINALS.contains(&id) {
            let all = id == "library.retrySidecars";
            self.check_originals(session, id, params, all)?;
        }
        Ok(())
    }

    /// An export (`app.export` after its preset is applied): its destination is required and must
    /// be inside a write root; `sameAsOriginal` writes next to the originals instead.
    pub fn check_export(&self, session: &Session, params: &Value) -> Result<(), String> {
        let same_as_original = params.get("destinationMode").and_then(Value::as_str) == Some("sameAsOriginal");
        let named = |k: &str| params.get(k).and_then(Value::as_str).is_some_and(|s| !s.trim().is_empty());
        if same_as_original && !named("path") {
            self.check_originals(session, "app.export", params, false)?;
        } else if !named("path") && !named("dir") {
            return Err("`app.export`: give `path` or `dir` (inside a write root) while file access is confined".into());
        }
        self.check_command(session, "app.export", params)
    }

    /// The photos `id` acts on (`ids` / `id`, else the selection, else the active photo; every
    /// photo with `all`) must have their originals inside a write root.
    fn check_originals(&self, session: &Session, id: &str, params: &Value, all: bool) -> Result<(), String> {
        let mut ids: Vec<PhotoId> = match (params.get("ids"), params.get("id")) {
            (Some(Value::Array(a)), _) => a.iter().filter_map(Value::as_u64).map(PhotoId).collect(),
            (_, Some(v)) => v.as_u64().map(PhotoId).into_iter().collect(),
            _ => Vec::new(),
        };
        if all {
            ids = session.catalog.photos().map(|p| p.id).collect();
        } else if ids.is_empty() {
            ids = session.selection.ids.clone();
            ids.extend(session.active());
        }
        for pid in ids {
            if let Some(photo) = session.catalog.photo(pid)
                && let Source::File { path } = &photo.source
            {
                self.check_path(&format!("`{id}`: photo {} (its original)", pid.0), path, Access::Write)?;
            }
        }
        Ok(())
    }

    /// `declared`: `key` is one of the command's file parameters, so a relative path is refused
    /// too (it would resolve against the server's working folder). Otherwise only absolute
    /// strings are checked, as reads, recursing into objects and arrays.
    fn check_value(&self, id: &str, key: &str, v: &Value, access: Access, declared: bool) -> Result<(), String> {
        match v {
            Value::String(s) => {
                if !declared && (NOT_PATHS.contains(&key) || !Path::new(s).is_absolute()) {
                    return Ok(());
                }
                if s.trim().is_empty() {
                    return Ok(());
                }
                self.check_path(&format!("`{id}`: `{key}`"), s, access)
            }
            Value::Array(items) => items.iter().try_for_each(|i| self.check_value(id, key, i, access, declared)),
            Value::Object(obj) => obj.iter().try_for_each(|(k, i)| self.check_value(id, k, i, Access::Read, false)),
            _ => Ok(()),
        }
    }

    /// Make the session's photo decoders read only inside the roots: a library may reference
    /// originals anywhere, and rendering one would hand its pixels to the agent.
    pub fn confine_loaders(self: &Arc<Self>, session: &mut Session) {
        let refuse = |path: &str| format!("{path} is outside the read or write roots; file access is confined");
        let media = &mut session.media;
        if let Some(inner) = media.file_loader.take() {
            let c = Arc::clone(self);
            media.file_loader = Some(Arc::new(move |path: &str, max: usize, with: lightcraft_engine::media::RawDecode<'_>| {
                if !c.allows(Path::new(path), Access::Read) {
                    return Err(refuse(path));
                }
                inner(path, max, with)
            }));
        }
        if let Some(inner) = media.reduced_loader.take() {
            let c = Arc::clone(self);
            media.reduced_loader = Some(Arc::new(move |path: &str, max: usize, with: lightcraft_engine::media::RawDecode<'_>| {
                if !c.allows(Path::new(path), Access::Read) {
                    return Err(refuse(path));
                }
                inner(path, max, with)
            }));
        }
        if let Some(inner) = media.file_probe.take() {
            let c = Arc::clone(self);
            media.file_probe = Some(Arc::new(move |path: &str| {
                if !c.allows(Path::new(path), Access::Read) {
                    return Err(refuse(path));
                }
                inner(path)
            }));
        }
        if let Some(inner) = media.preview_loader.take() {
            let c = Arc::clone(self);
            media.preview_loader =
                Some(Arc::new(move |path: &str, max: usize| if c.allows(Path::new(path), Access::Read) { inner(path, max) } else { None }));
        }
        if let Some(inner) = media.denoise.loader.take() {
            let c = Arc::clone(self);
            media.denoise.loader = Some(Arc::new(
                move |path: &str, max: usize, spec: &lightcraft_engine::media::DenoiseSpec, with: lightcraft_engine::media::RawDecode<'_>| {
                    if !c.allows(Path::new(path), Access::Read) {
                        return Err(refuse(path));
                    }
                    inner(path, max, spec, with)
                },
            ));
        }
    }
}

fn keys_for(table: &'static [(&'static str, &'static [&'static str])], id: &str) -> &'static [&'static str] {
    table.iter().find(|(c, _)| *c == id).map_or(&[], |(_, k)| k)
}

/// `path` made absolute and free of symbolic links as far as it exists: `None` for a relative
/// path or one with `..` (which could climb out of a root through a link).
fn resolve(path: &Path) -> Option<PathBuf> {
    if !path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir)) {
        return None;
    }
    // The deepest ancestor that exists, resolved; the rest (not yet created) appended as is.
    let mut existing = path.to_path_buf();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(c) = existing.canonicalize() {
            let mut out = c;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return Some(out);
        }
        rest.push(existing.file_name()?.to_os_string());
        if !existing.pop() {
            return None;
        }
    }
}
