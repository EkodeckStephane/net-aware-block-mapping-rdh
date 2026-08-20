$ErrorActionPreference = 'Stop'

# Rebuild the JVCIR revised LaTeX source package from the current paper tree.
# Run this script from the paper directory after pulling the revision branch.

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $paperDir

$zipPath = Join-Path $paperDir 'latex_source_v1_0_revised.zip'
$staging = Join-Path $env:TEMP ('rdh_latex_source_' + [Guid]::NewGuid().ToString('N'))

try {
    New-Item -ItemType Directory -Path $staging | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $staging 'images') | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $staging 'tracked_revision') | Out-Null

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
        if (-not (Test-Path $src)) {
            throw "Required source file is missing: $file"
        }
        Copy-Item $src (Join-Path $staging $file)
    }

    # Tracked-change fragments required by the blue and red-blue source wrappers.
    Get-ChildItem (Join-Path $paperDir 'tracked_revision') -File | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $staging 'tracked_revision' $_.Name)
    }

    # Only the figure PDFs actually referenced by the manuscript are needed.
    4..10 | ForEach-Object {
        $name = "Figure_$_.pdf"
        $src = Join-Path $paperDir 'images' $name
        if (-not (Test-Path $src)) {
            throw "Required figure is missing: images/$name"
        }
        Copy-Item $src (Join-Path $staging 'images' $name)
    }

    if (Test-Path $zipPath) {
        Remove-Item $zipPath -Force
    }

    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zipPath -CompressionLevel Optimal

    # Verify mandatory entries before reporting success.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        $names = $zip.Entries.FullName
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
        $zip.Dispose()
    }

    $hash = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $size = (Get-Item $zipPath).Length
    Write-Host "Created: $zipPath"
    Write-Host "Size:    $size bytes"
    Write-Host "SHA256:  $hash"
}
finally {
    if (Test-Path $staging) {
        Remove-Item $staging -Recurse -Force
    }
}
