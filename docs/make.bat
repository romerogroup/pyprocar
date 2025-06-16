@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.\source
set BUILDDIR=_build

REM Check for Sphinx installation
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

REM Check input arguments and provide custom behavior
if "%1" == "clean" (
	rmdir /s /q "%BUILDDIR%"
	rmdir /s /q "%SOURCEDIR%\examples"
	del errors.txt
	del sphinx_warnings.txt
	rmdir /s /q "%SOURCEDIR%\images\auto-generated"
	del "%SOURCEDIR%\getting-started\external_examples.rst"
	for /d /r %%x in (_autosummary) do (rmdir /s /q "%%x")
	goto end
)

if "%1" == "clean-except-examples" (
	rmdir /s /q "%BUILDDIR%"
	del errors.txt
	del sphinx_warnings.txt
	rmdir /s /q "%SOURCEDIR%\images\auto-generated"
	del "%SOURCEDIR%\getting-started\external_examples.rst"
	for /d /r %%x in (_autosummary) do (rmdir /s /q "%%x")
	goto end
)

if "%1" == "clean-autosummary" (
	for /d /r %%x in (_autosummary) do (rmdir /s /q "%%x")
	goto end
)

if "%1" == "phtml" (
	%SPHINXBUILD% -M html "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% -j auto
	goto end
)

if "%1" == "deploy" (
	:: Your deploy logic here. Remember that this script assumes it's being run with elevated permissions.
	:: You can mirror the logic in the Makefile but would need to adapt for Windows environment.
	@xcopy _build\html\* ..\docs\ /E /I /Y
	goto end
)

if "%1" == "html" (
    python "%SOURCEDIR%\generate_yml_docs.py"
    %SPHINXBUILD% -M html "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS%
	@xcopy _build\html\* ..\docs\ /E /I /Y
	rmdir /s /q _build\html
	rmdir /s /q _build\doctrees
	goto end
)


if "%1" == "pdf" (
    %SPHINXBUILD% -M latexpdf "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS%
    goto end
)

REM Default behavior for other cases
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd
