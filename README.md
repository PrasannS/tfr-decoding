Guide to all the code in this repo: 

SHP folder: contains notebooks exploring SteamSHP setting, quality, and outputs as suitable for our setting

src/
    models/ : code for TFR models, borrowed from previous project where we obtain TFR models
    tfr_decoding/ : code for monkey-patching huggingface code (some version dependence), in order to implement our sampling algorithms
    utils/ : code for helping with prefix sampling, processing data


analyze_entropy, directbeam, outspace_explore, prefix_explore: Notebooks for analysis on prefix decoding, which we generate results with

other files: for exploration with respect to SteamSHP setting, as well as for training of SteamSHP prefix models