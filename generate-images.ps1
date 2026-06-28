# Befor running the script execute:
# Set-ExecutionPolicy -Scope Process Bypass
#
# Example run:
#.\run_all.ps1 `
#    -InputFile "C:\temp\image.png" `
#    -OutputDir "C:\temp\outputdir"

param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

# Utwórz katalog wyjściowy, jeśli nie istnieje
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Nazwa pliku bez rozszerzenia
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)

$palettes = @(
    "popularity",
    "kmeans",
    "median_cut",
    "octree",
    "perceptual"
)

$dithers = @(
    "none",
    "floyd",
    "jarvis",
    "stucki",
    "sierra",
    "sierra_lite",
    "burkes",
    "atkinson",
    "bayer2",
    "bayer4",
    "bayer8",
    "adaptive"
)

$resizes = @(
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos"
)

$quantizers = @(
    "nearest",
    "perceptual",
    "weighted",
    "adaptive"
)

$total = $palettes.Count * $dithers.Count * $resizes.Count * $quantizers.Count
$i = 0

foreach ($palette in $palettes)
{
    foreach ($dither in $dithers)
    {
        foreach ($resize in $resizes)
        {
            foreach ($quantizer in $quantizers)
            {
                $i++

                Write-Host "[$i/$total] $palette | $dither | $resize | $quantizer"

                $OutputFile = Join-Path $OutputDir `
                    "$BaseName-p_${palette}-d_${dither}-r_${resize}-q_${quantizer}.png"

                python -m converter.main `
                    --palette $palette `
                    --dither $dither `
                    --resize $resize `
                    --quantizer $quantizer `
                    $InputFile `
                    $OutputFile
            }
        }
    }
}