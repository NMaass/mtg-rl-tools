# Run the Arena -> XMage mirror (after scripts\setup-arena-mirror.ps1).
#
#   scripts\arena-mirror.ps1 live [extra args...]
#       Launches the XMage display, follows the live MTGA Player.log, and
#       records a replay bundle under arena-mirror-runs\<timestamp>.
#       Useful extras: --from-start, --no-display, --log <path>, --out <dir>
#
#   scripts\arena-mirror.ps1 replay <bundle-dir> [extra args...]
#       Plays a recorded bundle back into the XMage display.
#       Useful extras: --step, --speed 2, --no-display

param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Mode,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest,
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
    python -m magic_cabt.arena_mirror $Mode --java "$JavaHome\bin\java.exe" @Rest
} finally { Pop-Location }
