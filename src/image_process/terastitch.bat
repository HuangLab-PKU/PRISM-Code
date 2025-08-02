@echo off

:: An example of stitching pipeline

:: 0] (OPTIONAL) Testing the reference system by visual inspection of the middle slice stitched using nominal stage coordinates
echo Testing the given reference system and stitching the middle slice of the volume using nominal stage coordinates...
echo.
terastitcher --test --volin="%CD%" --ref1=1 --ref2=2 --ref3=3 --vxl1=0.195 --vxl2=0.195 --vxl3=0.8
SET /P continue="Is the result correct (y/n)?"
IF NOT %continue% == y EXIT

:: 1] (OPTIONAL) Importing volume to the xml project file "xml_import" 
terastitcher --import --volin="%CD%" --ref1=1 --ref2=2 --ref3=3 --vxl1=0.195 --vxl2=0.195 --vxl3=0.8 --projout=xml_import

:: 2] Pairwise stacks displacement computation step. Needs a valid XML project file, which can (but does not need to) be produced by step 1]
terastitcher --displcompute --projin="%CD%/xml_import.xml" --projout=xml_displcomp --subvoldim=100

:: 3] Displacement projection
terastitcher --displproj --projin="%CD%/xml_displcomp.xml" --projout=xml_displproj

:: 4] Displacement thresholding
terastitcher --displthres --projin="%CD%/xml_displproj.xml" --projout=xml_displthres --threshold=0.7

:: 5] Optimal tiles placement
terastitcher --placetiles --projin="%CD%/xml_displthres.xml" --projout=xml_merging

:: Creating the directory where to put the stitching result
SET WORKINGDIR=%CD%
for %%* in (.) do SET DIRNAME=%%~n*
echo %DIRNAME%
cd..
mkdir %DIRNAME%_stitched
cd %DIRNAME%_stitched

:: 6] Merging and saving into a multiresolution representation (six resolutions are produced in this case)
terastitcher --merge --projin="%WORKINGDIR%/xml_merging.xml" --volout="%CD%/" --resolutions=012345
PAUSE