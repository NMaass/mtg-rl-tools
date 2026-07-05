# One-time setup for the Arena -> XMage mirror.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup-arena-mirror.ps1 `
#       [-XmageDir C:\Users\...\xmage-goldflush] [-SkipBuild]
#
# Steps: copy this overlay into the XMage checkout, build it, assemble the
# display's runtime classpath, and prewarm XMage's card database. Requires
# JDK 17 + Maven (see -JavaHome / -MavenHome defaults below).

param(
    [string]$XmageDir = "C:\Users\nicho\Code\xmage-goldflush",
    [string]$JavaHome = "C:\Users\nicho\tools\jdk-17.0.19+10",
    [string]$MavenHome = "C:\Users\nicho\tools\apache-maven-3.9.9",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$overlay = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $XmageDir "pom.xml"))) {
    throw "No XMage checkout at $XmageDir - clone https://github.com/magefree/mage there first."
}

Write-Host "== Copying overlay sources into $XmageDir"
foreach ($module in @("Mage.Server.Plugins\Mage.Player.AI", "Mage.Client")) {
    $src = Join-Path $overlay "$module\src"
    $dst = Join-Path $XmageDir "$module\src"
    if (Test-Path $src) {
        robocopy $src $dst /E /NJH /NJS /NDL /NFL | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $module" }
    }
}

$env:JAVA_HOME = $JavaHome
$env:Path = "$JavaHome\bin;$MavenHome\bin;" + $env:Path

if (-not $SkipBuild) {
    Write-Host "== Building XMage (skip tests; first build takes ~20+ minutes)"
    Push-Location $XmageDir
    try {
        & mvn -q -DskipTests "-Dcheckstyle.skip=true" install
        if ($LASTEXITCODE -ne 0) { throw "maven build failed" }
    } finally { Pop-Location }
}

Write-Host "== Assembling display runtime classpath"
Push-Location $XmageDir
try {
    & mvn -q -pl Mage.Client org.apache.maven.plugins:maven-dependency-plugin:3.6.1:build-classpath `
        "-Dmdep.outputFile=target/mirror-deps.txt" "-Dcheckstyle.skip=true"
    if ($LASTEXITCODE -ne 0) { throw "classpath assembly failed" }
    $deps = Get-Content (Join-Path $XmageDir "Mage.Client\target\mirror-deps.txt") -Raw
    $cp = "$XmageDir\Mage.Client\target\classes;$XmageDir\Mage.Sets\target\mage-sets.jar;$deps"
    Set-Content -Path (Join-Path $XmageDir "Mage.Client\target\mirror-classpath.txt") -Value $cp -Encoding ascii -NoNewline
} finally { Pop-Location }

Write-Host "== Prewarming XMage card database (one-time, a few minutes)"
Push-Location (Join-Path $XmageDir "Mage.Client")
try {
    & "$JavaHome\bin\java.exe" -Xmx2g "-Dfile.encoding=UTF-8" -cp $cp mage.client.cabtmirror.ArenaMirrorApp --prewarm
    if ($LASTEXITCODE -ne 0) { throw "card database prewarm failed" }
} finally { Pop-Location }

Write-Host ""
Write-Host "Setup complete. Run a live session with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\arena-mirror.ps1 live"
