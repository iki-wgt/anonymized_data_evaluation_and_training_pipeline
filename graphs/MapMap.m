%%==========================================================================
% MapMap
%
% A lightweight container class for storing and filtering string-based 
% entries annotated by categorical metadata (size, frequency). 
% Designed for efficient grouping and retrieval in experimental pipelines 
% involving image attributes or class subsets.
%
% Usage:
%   mm = MapMap();
%   mm.add(data, 'size', 'medium', 'frequency', 'high');
%   out = mm.get('size','medium','frequency','high');
%
% Properties:
%   - size         (string array)  % Category label for entry size
%   - frequency    (string array)  % Category label for entry frequency
%   - entry        (string array)  % Stored data entries
%   - additional   (string array)  % Reserved for future use
%
% Methods:
%   - add(data, options)
%       Appends new entries with associated metadata
%
%   - get(options)
%       Retrieves entries by filtering based on `size` and `frequency`
%
% Dependencies:
%   - None (pure MATLAB OOP)
%
% Notes:
%   - Filtering supports 'all' as a wildcard for both attributes
%   - Class extends `handle`, enabling mutable behavior
%
% Author:    Sarah Weiß  
%            @ Institute for Artificial Intelligence,  
%              Ravensburg-Weingarten University of Applied Sciences  
%            @ [https://github.com/iki-wgt](https://github.com/iki-wgt) or [https://github.com/Fox93](https://github.com/Fox93)
% Date:      03/15/2025
%==========================================================================

classdef MapMap < handle

  properties
    size = strings(0);         % Size category for each entry
    frequency = strings(0);    % Frequency category for each entry
    entry = strings(0);        % Data entries
    additional = strings(0);   % Reserved for future use
  end

  methods

    function obj = MapMap()
        % Constructor: Initializes empty object
    end

    function data_out = get(obj, options)
      % Retrieve entries matching specified size and frequency
      arguments
        obj 
        options.size = 'all'
        options.frequency = 'all'
      end

      % Default to all entries
      s_idx = true(size(obj.size,1),1);
      f_idx = true(size(obj.size,1),1);

      % Apply size filter
      if ~strcmp(options.size, 'all')
        s_idx = obj.size == options.size;
      end

      % Apply frequency filter
      if ~strcmp(options.frequency, 'all')
        f_idx = obj.frequency == options.frequency;
      end

      % Logical intersection
      total_idx = f_idx & s_idx;

      % Return filtered entries
      data_out = obj.entry(total_idx);
    end

    function add(obj, data, options)
      % Append entries with associated size and frequency labels
      arguments
        obj
        data                     % String array or compatible input
        options.frequency        % Label for frequency
        options.size             % Label for size
      end

      % Compute range for insertion
      n_elems = numel(data);
      start_elem = numel(obj.entry);

      % Assign entries and metadata
      obj.entry(start_elem+1:start_elem+n_elems) = data;
      obj.size(start_elem+1:start_elem+n_elems) = options.size;
      obj.frequency(start_elem+1:start_elem+n_elems) = options.frequency;
    end

  end
end
