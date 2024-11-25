*********************
Configuration options
*********************

There are many configuration options that can be adjusted to change the behaviour of the roocs stack.
The configuration file used can always be found under ``<package>/etc/roocs.ini`` where package is a package in `roocs` e.g. `roocs-utils`.

Any section of the configuration files can be overwritten by creating a new INI file with the desired sections and values and then setting the environment variable ``ROOCS_CONFIG`` as the file path to the new INI file.
e.g. ``ROOCS_CONFIG="path/to/config.ini"``

Below are the configuration settings, with explanations provided as comments within the code blocks. The examples given may not directly correspond to the settings used in each package.

Specifying types
################

It is possible to specify the type of the entries in the configuration file, for example if you want a value to be a list when the file is parsed.

This is managed through a ``[config_data_types]`` section at the top of the ``ini`` file which has the following options:

# fmt: off
.. code-block::

    [config_data_types]
    lists =
    dicts =
    ints =
    floats =
    boolean =
    extra_lists =
    extra_dicts =
    extra_ints =
    extra_floats =
    extra_booleans =

# fmt: on

Simply adding the name of the value you want to format after ``=`` will render in the correct format. e.g. ``boolean = use_inventory is_default_for_path`` will set both ``use_inventory`` and ``is_default_for_path`` as booleans.

roocs-utils
###########

In roocs-utils there are project level settings. The settings under each project heading are the same.
e.g. for cmip5 the heading is ``[project:cmip5]``:

.. code-block::

    [project:cmip5]
    project_name = cmip5
    # base directory for data file paths
    base_dir = /badc/cmip5/data/cmip5
    # if a dataset id is identified as coming from this project, should these be the default settings used (as opposed to using the c3s-cmip5 settings by default)
    is_default_for_path = True
    # template for the output file name - used in ``clisops.utils.file_namers``
    file_name_template = {__derive__var_id}_{frequency}_{model_id}_{experiment_id}_r{realization}i{initialization_method}p{physics_version}{__derive__time_range}{extra}.{__derive__extension}
    # defaults used in file name template above if the dataset doesn't contain the attribute
    attr_defaults =
        model_id:no-model
        frequency:no-freq
        experiment:no-expt
        realization:X
        initialization_method:X
        physics_version:X
    # the order of facets in the file paths of datasets for this project
    facet_rule = activity product institute model experiment frequency realm mip_table ensemble_member version variable
    # what particular facets will be identified as in this project - not currently used
    mappings =
        project:project_id
    # whether to use an intake catalog or not for this project
    use_catalog = False
    # where original files can be downloaded
    data_node_root = https://data.mips.climate.copernicus.eu/thredds/fileServer/esg_c3s-cmip6/


There are settings for the environment:

.. code-block::

    [environment]
    # relating to the number of threads to use for processing
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS = 1
    NUMEXPR_NUM_THREADS = 1

The elastic search settings are specified here:

.. code-block::

    [elasticsearch]
    endpoint = elasticsearch.ceda.ac.uk
    port = 443
    # names of the elasticsearch indexes used for the various stores
    character_store = roocs-char
    fix_store = roocs-fix
    analysis_store = roocs-analysis
    fix_proposal_store = roocs-fix-prop


clisops
#######

These are settings that are specific to `clisops`:

.. code-block::

    [clisops:read]
    # memory limit for chunks - dask breaks up its underlying array into chunks
    chunk_memory_limit = 250MiB

    [clisops:write]
    # maximum file size of output files. Files are split if this is exceeded
    file_size_limit = 1GB
    # staging directory to output files to before they are moved to the requested output directory
    # if unset, the files are output straight to the requested output directory
    output_staging_dir = /gws/smf/j04/cp4cds1/c3s_34e/rook_prod_cache


daops
#####

`daops` provides settings for using the `intake` catalog:

.. code-block::

    [catalog]
    # provides the url for the intake catalog with details of datasets
    intake_catalog_url = https://raw.githubusercontent.com/cp4cds/c3s_34g_manifests/master/intake/catalogs/c3s.yaml


rook
####

There are currently no settings in `rook` but these would be set in the same way as the `clisops` and `daops` settings. e.g. with ``[rook:section]`` headings.

dachar
######

These are settings that are specific to `dachar`:

.. code-block::

    [dachar:processing]
    # LOTUS settings for scanning datasets
    queue = short-serial
    # large settings for scanning large datasets
    wallclock_large = 23:59
    memory_large = 32000
    # settings for scanning smaller datasets
    wallclock_small = 04:00
    memory_small = 4000

    [dachar:output_paths]
    # output paths for scanning datasets and generating fixes
    _base_path = ./outputs
    base_log_dir = %(_base_path)s/logs
    batch_output_path = %(base_log_dir)s/batch-outputs/{grouped_ds_id}
    json_output_path = %(_base_path)s/register/{grouped_ds_id}.json
    success_path = %(base_log_dir)s/success/{grouped_ds_id}.log
    no_files_path = %(base_log_dir)s/failure/no_files/{grouped_ds_id}.log
    pre_extract_error_path = %(base_log_dir)s/failure/pre_extract_error/{grouped_ds_id}.log
    extract_error_path = %(base_log_dir)s/failure/extract_error/{grouped_ds_id}.log
    write_error_path = %(base_log_dir)s/failure/write_error/{grouped_ds_id}.log
    fix_path = %(_base_path)s/fixes/{grouped_ds_id}.json


    [dachar:checks]
    # checks to run when analysing a sample of datasets
    # common checks are run on all samples
    common = coord_checks.RankCheck coord_checks.MissingCoordCheck
    # it is possible to specify checks that will be run on datasets from specific projects
    cmip5 =
    cmip6 =
    cordex = coord_checks.ExampleCheck


    [dachar:settings]
    # elasticsearch api token that allows write access to indexes
    elastic_api_token =
    # how many directories levels to join by to create the name of a new directory when outputting results of scans
    # see ``dachar.utils.switch_ds.get_grouped_ds_id``
    dir_grouping_level = 4
    # threshold at which an anomaly in a sample of datasets will be identified for a fix - not currently used
    # the lower threshold (between 0 and 1), the more likely the anomaly will be to get fixed
    concern_threshold = 0.2
    # possible locations for scans and analysis of datasets
    locations = ceda dkrz other


catalog maker
#############

In the catalog maker there are project level settings as well.
The settings under each project heading are the same.
Settings for the catalog maker are:

.. code-block::

    [project:c3s-cmip6]
    # directory to store catalog and dataset list used in generation of catalog
    # if catalog_dir is the same for different projects, the yaml file in this directory will be updated for each project, rather than a new one made
    catalog_dir = ./catalog_data
    # Where the csv file will be generated
    csv_dir = %(catalog_dir)s/%(project_name)s/
    # Where the user will provide a dataset list which will be used to generate the catalog
    datasets_file = %(catalog_dir)s/%(project_name)s-datasets.txt

Further settings for the intake catalog workflow are::

    [log]
    # directory for logging outputs from LOTUS when generating catalog entries
    log_base_dir = /gws/smf/j04/cp4cds1/c3s_34e/inventory/log

    [workflow]
    split_level = 4
    # max duration for LOTUS jobs, as "hh:mm:ss"
    max_duration = 04:00:00
    # job queue on LOTUS
    job_queue = short-serial
    # number of datasets to process in one batch - fewer batches is better as it prevents "Exception: Could not obtain file lock" error
    n_per_batch = 750
