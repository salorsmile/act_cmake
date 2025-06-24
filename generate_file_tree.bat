@echo off
setlocal enabledelayedexpansion

echo Directory Tree of %CD%
echo.

:: 递归函数用于打印目录树
:print_tree
set "dir=%~1"
set "prefix=%~2"

:: 打印当前目录
echo %prefix%%~nx1\

:: 获取子目录
for /d %%d in ("%dir%\*") do (
    set "subdir=%%d"
    call :print_tree "!subdir!" "  %prefix%"
)

:: 获取文件
for %%f in ("%dir%\*.*") do (
    echo %prefix%  %%~nxf
)

goto :eof

:: 主程序
call :print_tree "%CD%" ""
pause