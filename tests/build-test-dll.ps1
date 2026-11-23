param([Parameter(Mandatory)][string]$Work, [Parameter(Mandatory)][string]$Destination)
$ErrorActionPreference = 'Stop'
$src = Join-Path $Work 'gufo'
$libDir = Join-Path $Work 'build/extensions/build/qwen38_flash_next/build'
$llvm = Join-Path $env:HIP_PATH 'lib/llvm/bin'
$out = [IO.Directory]::CreateDirectory($Destination).FullName
$obj = Join-Path $out 'test_exports.obj'
& "$llvm/clang-cl.exe" /nologo /c /MD /std:c++20 /EHsc /D__HIP_PLATFORM_AMD__ "/I$src" "/I$src/include" "/I$env:HIP_PATH/include" "/Fo$obj" "$PSScriptRoot/test_exports.cpp"
if ($LASTEXITCODE) { throw 'Compiling test exports failed.' }
$dll = Join-Path $out 'mixed-test.dll'
& "$llvm/lld-link.exe" /DLL /NOLOGO /OPT:REF /OPT:ICF "/OUT:$dll" $obj "$libDir/gufo_qwen38_flash_next_hip.lib" "$libDir/gufo_qwen38_flash_next_mmq.lib" "$env:HIP_PATH/lib/amdhip64.lib" /EXPORT:qfn_mmq_init /EXPORT:qfn_mmq_mixed_moe_raw /EXPORT:qfn_mmq_moe_vec
if ($LASTEXITCODE) { throw 'Linking candidate kernel objects failed.' }
Write-Output $dll
