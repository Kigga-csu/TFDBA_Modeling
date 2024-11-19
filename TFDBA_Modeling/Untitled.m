clc
clear

clear classes
%obj = py.importlib.import_module('ModelTrain');
%py.importlib.reload(obj);

py.ModelTrain.test()

