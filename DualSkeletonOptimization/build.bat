@echo off

rmdir /s /q build
mkdir build
cd build


cmake .. -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake


cmake --build . --config Release

move "%~dp0U_tube.txt" "%~dp0build\" >nul 2>&1

pause