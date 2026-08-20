$ErrorActionPreference = 'Stop'

# Rebuild the JVCIR revised LaTeX source package from the current paper tree.
# Compatible with Windows PowerShell 5.1 and PowerShell 7+.
# Run this script from the paper directory after pulling the revision branch.

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $paperDir

$zipPath = Join-Path $paperDir 'latex_source_v1_0_revised.zip'
$staging = Join-Path $env:TEMP ('rdh_latex_source_' + [Guid]::NewGuid().ToString('N'))
$stagingImages = Join-Path $staging 'images'
$stagingTracked = Join-Path $staging 'tracked_revision'
$paperImages = Join-Path $paperDir 'images'
$paperTracked = Join-Path $paperDir 'tracked_revision'

try {
    New-Item -ItemType Directory -Path $staging | Out-Null
    New-Item -ItemType Directory -Path $stagingImages | Out-Null
    New-Item -ItemType Directory -Path $stagingTracked | Out-Null

    # Clean manuscript and Elsevier support files.
    $rootFiles = @(
        'main.tex',
        'references.bib',
        'cas-dc.cls',
        'cas-common.sty',
        'cas-model1-num-names.bst',
        'graphical_abstract.tex',
        'cover_letter.tex',
        'revision_cover_letter.tex',
        'response_to_reviewers.tex',
        'highlights.txt',
        'figure_captions_and_alt_text.txt',
        'article_v1_0_blue_changes.tex',
        'article_v1_0_red_blue_changes.tex'
    )

    foreach ($file in $rootFiles) {
        $src = Join-Path $paperDir $file
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Required source file is missing: $file"
        }
        $dst = Join-Path $staging $file
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }

    # Tracked-change fragments required by the blue and red-blue source wrappers.
    if (-not (Test-Path -LiteralPath $paperTracked)) {
        throw 'Required directory is missing: tracked_revision'
    }
    Get-ChildItem -LiteralPath $paperTracked -File | ForEach-Object {
        $dst = Join-Path $stagingTracked $_.Name
        Copy-Item -LiteralPath $_.FullName -Destination $dst -Force
    }

    # Only the figure PDFs actually referenced by the manuscript are needed.
    foreach ($number in 4..10) {
        $name = "Figure_$number.pdf"
        $src = Join-Path $paperImages $name
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Required figure is missing: images/$name"
        }
        $dst = Join-Path $stagingImages $name
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }

    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zipPath -CompressionLevel Optimal -Force

    # Verify mandatory entries before reporting success.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        $names = @($zip.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
        $mandatory = @(
            'main.tex',
            'references.bib',
            'cas-dc.cls',
            'cas-common.sty',
            'cas-model1-num-names.bst',
            'article_v1_0_blue_changes.tex',
            'article_v1_0_red_blue_changes.tex',
            'tracked_revision/common_01.tex',
            'tracked_revision/common_07.tex',
            'images/Figure_4.pdf',
            'images/Figure_10.pdf'
        )
        foreach ($entry in $mandatory) {
            if ($names -notcontains $entry) {
                throw "ZIP verification failed: missing $entry"
            }
        }
    }
    finally {
        if ($null -ne $zip) {
            $zip.Dispose()
        }
    }

    $hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $size = (Get-Item -LiteralPath $zipPath).Length
    Write-Host "Created: $zipPath"
    Write-Host "Size:    $size bytes"
    Write-Host "SHA256:  $hash"
}
finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
