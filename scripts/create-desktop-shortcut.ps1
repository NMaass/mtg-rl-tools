# Create a Desktop shortcut named "Arena Mirror" that launches the mirror GUI.
#
#   powershell -ExecutionPolicy Bypass -File scripts\create-desktop-shortcut.ps1
#
# One-time convenience: afterwards, double-click "Arena Mirror" on your Desktop
# instead of running any command.

$ErrorActionPreference = "Stop"
$overlay = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $overlay "Arena Mirror.bat"
if (-not (Test-Path $launcher)) {
    throw "Launcher not found at $launcher"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Arena Mirror.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.WorkingDirectory = $overlay
$shortcut.Description = "Follow MTG Arena and mirror it into XMage"
# a recognizable icon; falls back silently if the exe is absent
$py = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if ($py) { $shortcut.IconLocation = "$py,0" }
$shortcut.Save()

Write-Host "Created shortcut: $shortcutPath"
