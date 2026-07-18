# mtgo-video-acquisition

Creates a reproducible local corpus manifest for MTGO videos and optionally
invokes `yt-dlp`. It never logs into a site, bypasses DRM, or supplies cookies.
Download mode requires an explicit acknowledgement that you have permission and
that the download complies with the source platform's terms.

```powershell
mtgo-video-source discover --url PLAYLIST_OR_CHANNEL_URL --out sources.json
mtgo-video-source download --manifest sources.json --out D:\mtgo-corpus `
  --acknowledge-rights-and-terms
mtgo-video-source import-local --directory D:\videos --out local-sources.json
```

The output manifest records source URL, extractor ID, title, channel/uploader,
duration, upload date, local file path, file size, and SHA-256. The parser uses
local files only, so the acquisition policy can be changed independently.
