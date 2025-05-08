%%==========================================================================
% basicExportSVG
%
% Utility function to export a MATLAB figure as an SVG (Scalable Vector 
% Graphics) file. Automatically appends '.svg' extension if absent.
%
% Usage:
%   basicExportSVG(figH, fname)
%   basicExportSVG(figH, fname, options)
%
% Inputs:
%   - figH           % Figure handle to be exported
%   - fname (char)   % Target file name (with or without '.svg' extension)
%   - options.format (string, optional)
%                   % Placeholder for future extension (e.g., alternate drivers)
%
% Outputs:
%   - <fname>.svg    % Vector-format figure (SVG)
%
% Dependencies:
%   - None (uses built-in `print` function with '-dsvg' driver)
%
% Notes:
%   - Output is optimized for compatibility with vector-based tools (e.g., Inkscape, Illustrator)
%   - File naming is automatically sanitized to include `.svg` suffix if omitted
%
% Author:    Sarah Weiß
%            @ Institute for Artificial Intelligence,
%              Ravensburg-Weingarten University of Applied Sciences
%            @ [https://github.com/iki-wgt](https://github.com/iki-wgt) or [https://github.com/Fox93](https://github.com/Fox93)
% Date:      03/15/2025
%==========================================================================

function basicExportSVG(figH, fname, options)
    arguments
        figH                    % Figure handle
        fname char              % Output file name
        options.format string = "" % Placeholder argument
    end

    % Ensure filename has '.svg' extension
    if ~contains(fname, '.svg')
        fname = [fname '.svg'];
    end

    % Export using SVG driver
    print(figH, fname, '-dsvg');
end