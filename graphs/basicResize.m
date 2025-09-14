%%==========================================================================
% basicResize
%
% Utility function to standardize figure dimensions in centimeters for 
% consistent export and publication layout. Ensures fixed width-height 
% ratio across figures, matching A4-compatible widths by default.
%
% Usage:
%   basicResize(figH)
%   basicResize(figH, options)
%
% Inputs:
%   - figH               % Figure handle to be resized
%   - options.height     % Target height in cm (default: 5.1)
%   - options.widthA4    % Target width in cm (default: 14.4)
%
% Outputs:
%   - Directly modifies the size of the provided figure handle
%
% Dependencies:
%   - None (pure MATLAB graphics manipulation)
%
% Notes:
%   - All units are internally converted to centimeters
%   - Width of 14.4 cm corresponds to full-width A4 compatibility
%
% Author:    Sarah Weiß  
%            @ Institute for Artificial Intelligence,  
%              Ravensburg-Weingarten University of Applied Sciences  
%            @ https://github.com/iki-wgt or https://github.com/Fox93
% Date:      03/15/2025
%==========================================================================

function basicResize(figH, options)
    arguments
        figH                            % Figure handle
        options.height double = 5.1     % Desired height in cm
        options.widthA4 = 14.4          % Desired width in cm (A4-compatible)
    end

    figH.Units = "centimeters";
    figH.Position(3:4) = [options.widthA4 options.height];
end
