# Launch the Arena -> XMage mirror GUI.
#
#   scripts\arena-mirror-gui.ps1        (or double-click "Arena Mirror.bat")
#
# Opens a window with a "Locate MTGA logs" button, live log + actions panes,
# and Start/Stop. On Start it follows the Player.log and, when a live game
# appears, launches XMage and mirrors the current game while recording a
# CABT bundle under the chosen output folder.
#
# If XMage has not been built yet (scripts\setup-arena-mirror.ps1), the GUI
# still opens and records — it just can't show the XMage board until built.

param(
    [string]$XmageDir = "C:\Users\nicho\Code\xmage-goldflush",
    [string]$JavaHome = "C:\Users\nicho\tools\jdk-17.0.19+10"
)

$ErrorActionPreference = "Stop"
$overlay = Split-Path -Parent $PSScriptRoot

# Locate a Python interpreter.
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    $python = (Get-Command py -ErrorAction SilentlyContinue).Source
}
if (-not $python) {
    throw "Python was not found on PATH. Install Python 3, then retry."
}

# Wire up the XMage display if it has been built; otherwise run record-only.
$classpathFile = Join-Path $XmageDir "Mage.Client\target\mirror-classpath.txt"
$javaExe = Join-Path $JavaHome "bin\java.exe"
$guiArgs = @("-m", "magic_cabt.arena_mirror", "gui")
$workDir = $overlay
if (Test-Path $classpathFile) {
    $env:MAGIC_CABT_CLASSPATH = (Get-Content $classpathFile -Raw).Trim()
    if (Test-Path $javaExe) {
        $guiArgs += @("--java", $javaExe)
    }
    # the display JVM must run from Mage.Client so XMage finds its db/ + images
    $workDir = Join-Path $XmageDir "Mage.Client"
    Write-Host "XMage display: ready."
} else {
    Write-Host "XMage is not built yet - the GUI will record only."
    Write-Host "To enable the live board, run once:"
    Write-Host "    scripts\setup-arena-mirror.ps1"
}

$env:PYTHONPATH = (Join-Path $overlay "python") + ";" + $env:PYTHONPATH

Push-Location $workDir
try {
    & $python @guiArgs
} finally { Pop-Location }
