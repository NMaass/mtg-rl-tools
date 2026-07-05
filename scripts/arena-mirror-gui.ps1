# Launch the Arena -> XMage mirror GUI (after scripts\setup-arena-mirror.ps1).
#
#   scripts\arena-mirror-gui.ps1
#
# Opens a window with a "Locate MTGA logs" button, live log + actions panes,
# and Start/Stop. On Start it follows the Player.log and, when a live game
# appears, launches XMage and mirrors the current game while recording a
# CABT bundle under the chosen output folder.

param(
    [string]$XmageDir = "C:\Users\nicho\Code\xmage-goldflush",
    [string]$JavaHome = "C:\Users\nicho\tools\jdk-17.0.19+10"
)

$ErrorActionPreference = "Stop"
$overlay = Split-Path -Parent $PSScriptRoot

$classpathFile = Join-Path $XmageDir "Mage.Client\target\mirror-classpath.txt"
if (-not (Test-Path $classpathFile)) {
    throw "No $classpathFile - run scripts\setup-arena-mirror.ps1 first."
}

$env:MAGIC_CABT_CLASSPATH = (Get-Content $classpathFile -Raw).Trim()
$env:PYTHONPATH = (Join-Path $overlay "python") + ";" + $env:PYTHONPATH

# the display JVM must run from Mage.Client so XMage finds its db/ and images
Push-Location (Join-Path $XmageDir "Mage.Client")
try {
    python -m magic_cabt.arena_mirror gui --java "$JavaHome\bin\java.exe"
} finally { Pop-Location }
