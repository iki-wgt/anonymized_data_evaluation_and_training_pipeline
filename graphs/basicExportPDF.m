%%==========================================================================
% basicExportPDF
%
% Utility function to export a MATLAB figure as a vector-based PDF and 
% simultaneously save the figure in MATLAB .fig format.
%
% Usage:
%   basicExportPDF(figH, fname)
%   basicExportPDF(figH, fname, options)
%
% Inputs:
%   - figH           % Figure handle to be exported
%   - fname (char)   % Target file name (with or without '.pdf' extension)
%   - options.format (string, optional)
%                   % Placeholder for future extension (e.g., 'raster')
%
% Outputs:
%   - <fname>.pdf    % Vector-format figure
%   - <fname>.fig    % MATLAB figure file (same base name)
%
% Dependencies:
%   - None (uses built-in exportgraphics and savefig)
%
% Notes:
%   - Output is enforced as PDF vector graphics
%   - Appends '.pdf' if missing in provided file name
%
% Author:    Sarah Weiß
%            @ Institute for Artificial Intelligence,
%              Ravensburg-Weingarten University of Applied Sciences
%            @ https://github.com/iki-wgt or https://github.com/Fox93
% Date:      03/15/2025
%==========================================================================

function basicExportPDF(figH,fname,options)
	arguments
		figH                        % Figure handle
		fname char                  % Output file name
		options.format string = ""  % Placeholder argument
	end

    % Ensure filename has '.pdf' extension
	if ~contains(fname,'.pdf')
		fname = [fname '.pdf'];
	end

    % Export as vector-based PDF
	exportgraphics(figH,fname,'ContentType','vector')
	savefig(strrep(fname,'.pdf',''))
end