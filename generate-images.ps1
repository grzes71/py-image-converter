# Befor running the script execute:
# Set-ExecutionPolicy -Scope Process Bypass
#
# Example run:
#.\run_all.ps1 `
#    -InputFile "C:\temp\image.png" `
#    -OutputDir "C:\temp\outputdir"
# Try p_octree / d_bayer2 / r_nearest / q_adaptive,q_weighted,q_perceptual,q_nearest for a good result
# Try p_median_cut / d_bayer2 / r_bicubic / q_perceptual for a good result

param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [int]$Height,

    [Parameter(Mandatory = $true)]
    [int]$Width
)

$env:PYTHONPATH = "src"

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
    "adaptive",
    "atari_false_floyd",
    "atari_two_row_sierra",
    "atari_diagonal",
    "atari_bayer2v2",
    "atari_bayer2x4",
    "atari_non_rectangular8"
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
                    --height $Height `
                    --width $Width `
                    $InputFile `
                    $OutputFile
            }
        }
    }
}