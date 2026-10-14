from distutils.command.build import build
from distutils.command.clean import clean
from distutils.version import LooseVersion
import os
import platform
import re
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CopyPreBuild(build):
    def initialize_options(self):
        build.initialize_options(self)
        # We just overwrite this because the default "build" (and "build/lib")
        # clashes with directories many developers have in their source trees;
        # this can create confusing results with "pip install .", which clones
        # the whole source tree by default
        self.build_base = '_tmppythonbuild'

    def run(self):
        # remove existing build directory
        #   by default, this stays around. we want to make sure generated
        #   files like libartemis.(2d|3d|rz).(so|pyd) are always only the
        #   ones we want to package and not ones from an earlier wheel's stage
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()

        # call superclass
        build.run(self)

        # matches: libartemis.(2d|3d|rz).(so|pyd)
        re_libprefix = re.compile(r"libartemis\...\.(?:so|dll)")
        libs_found = []
        for lib_name in os.listdir(PYARTEMIS_LIB_DIR):
            if re_libprefix.match(lib_name):
                lib_path = os.path.join(PYARTEMIS_LIB_DIR, lib_name)
                libs_found.append(lib_path)
        if len(libs_found) == 0:
            raise RuntimeError("Error: no pre-build Artemis libraries found in "
                               "PYARTEMIS_LIB_DIR='{}'".format(PYARTEMIS_LIB_DIR))

        # copy external libs into collection of files in a temporary build dir
        dst_path = os.path.join(self.build_lib, "pyartemis")
        for lib_path in libs_found:
            shutil.copy(lib_path, dst_path)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake 3.20.0+ must be installed to build the following " +
                "extensions: " +
                ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(
            r'version\s*([\d.]+)',
            out.decode()
        ).group(1))
        if cmake_version < '3.20.0':
            raise RuntimeError("CMake >= 3.20.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)
        ))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        r_dim = re.search(r'artemis_(1|2|3|rz)(?:d*)', ext.name)
        dims = r_dim.group(1).upper()

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
            os.path.join(extdir, "pyartemis"),
            '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + extdir,
            '-DArtemis_DIMS=' + dims,
            '-DArtemis_APP:BOOL=OFF',
            '-DArtemis_LIB:BOOL=ON',
            ## variants
            '-DArtemis_COMPUTE=' + ARTEMIS_COMPUTE,
            '-DArtemis_MPI:BOOL=' + ARTEMIS_MPI,
            '-DArtemis_EB:BOOL=' + ARTEMIS_EB,
            '-DArtemis_OPENPMD:BOOL=' + ARTEMIS_OPENPMD,
            '-DArtemis_PRECISION=' + ARTEMIS_PRECISION,
            '-DArtemis_PARTICLE_PRECISION=' + ARTEMIS_PARTICLE_PRECISION,
            '-DArtemis_PSATD:BOOL=' + ARTEMIS_PSATD,
            '-DArtemis_QED:BOOL=' + ARTEMIS_QED,
            '-DArtemis_QED_TABLE_GEN:BOOL=' + ARTEMIS_QED_TABLE_GEN,
            ## dependency control (developers & package managers)
            '-DArtemis_amrex_internal=' + ARTEMIS_AMREX_INTERNAL,
            #        see PICSAR and openPMD below
            ## static/shared libs
            '-DBUILD_SHARED_LIBS:BOOL=' + BUILD_SHARED_LIBS,
            ## Unix: rpath to current dir when packaged
            ##       needed for shared (here non-default) builds and ADIOS1
            ##       wrapper libraries
            '-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON',
            '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=OFF',
            # Windows: has no RPath concept, all `.dll`s must be in %PATH%
            #          or same dir as calling executable
        ]
        if ARTEMIS_QED.upper() in ['1', 'ON', 'TRUE', 'YES']:
            cmake_args.append('-DArtemis_picsar_internal=' + ARTEMIS_PICSAR_INTERNAL)
        if ARTEMIS_OPENPMD.upper() in ['1', 'ON', 'TRUE', 'YES']:
            cmake_args += [
                '-DHDF5_USE_STATIC_LIBRARIES:BOOL=' + HDF5_USE_STATIC_LIBRARIES,
                '-DADIOS_USE_STATIC_LIBS:BOOL=' + ADIOS_USE_STATIC_LIBS,
                '-DArtemis_openpmd_internal=' + ARTEMIS_OPENPMD_INTERNAL,
            ]
        # further dependency control (developers & package managers)
        if ARTEMIS_AMREX_SRC:
            cmake_args.append('-DArtemis_amrex_src=' + ARTEMIS_AMREX_SRC)
        if ARTEMIS_AMREX_REPO:
            cmake_args.append('-DArtemis_amrex_repo=' + ARTEMIS_AMREX_REPO)
        if ARTEMIS_AMREX_BRANCH:
            cmake_args.append('-DArtemis_amrex_branch=' + ARTEMIS_AMREX_BRANCH)
        if ARTEMIS_OPENPMD_SRC:
            cmake_args.append('-DArtemis_openpmd_src=' + ARTEMIS_OPENPMD_SRC)
        if ARTEMIS_PICSAR_SRC:
            cmake_args.append('-DArtemis_picsar_src=' + ARTEMIS_PICSAR_SRC)
        if ARTEMIS_CCACHE_PROGRAM is not None:
            cmake_args.append('-DCCACHE_PROGRAM=' + ARTEMIS_CCACHE_PROGRAM)

        if sys.platform == "darwin":
            cmake_args.append('-DCMAKE_INSTALL_RPATH=@loader_path')
        else:
            # values: linux*, aix, freebsd, ...
            #   just as well win32 & cygwin (although Windows has no RPaths)
            cmake_args.append('-DCMAKE_INSTALL_RPATH=$ORIGIN')

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(),
                    os.path.join(extdir, "pyartemis")
                )
            ]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        build_args += ['--parallel', BUILD_PARALLEL]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        build_dir = os.path.join(self.build_temp, dims)
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=build_dir,
            env=env
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_dir
        )
        # note that this does not call install;
        # we pick up artifacts directly from the build output dirs


with open('./README.md', encoding='utf-8') as f:
    long_description = f.read()

# Allow to control options via environment vars.
#   Work-around for https://github.com/pypa/setuptools/issues/1712
# Pick up existing Artemis libraries or...
PYARTEMIS_LIB_DIR = os.environ.get('PYARTEMIS_LIB_DIR')

env = os.environ.copy()
# ... build Artemis libraries with CMake
#   note: changed default for SHARED, MPI, TESTING and EXAMPLES
#   note: we use all-uppercase variable names for environment control to be
#         consistent across platforms (especially Windows)
ARTEMIS_COMPUTE = env.pop('ARTEMIS_COMPUTE', 'OMP')
ARTEMIS_MPI = env.pop('ARTEMIS_MPI', 'OFF')
ARTEMIS_EB = env.pop('ARTEMIS_EB', 'OFF')
ARTEMIS_OPENPMD = env.pop('ARTEMIS_OPENPMD', 'ON')
ARTEMIS_PRECISION = env.pop('ARTEMIS_PRECISION', 'DOUBLE')
ARTEMIS_PARTICLE_PRECISION = env.pop('ARTEMIS_PARTICLE_PRECISION', ARTEMIS_PRECISION)
ARTEMIS_PSATD = env.pop('ARTEMIS_PSATD', 'OFF')
ARTEMIS_QED = env.pop('ARTEMIS_QED', 'ON')
ARTEMIS_QED_TABLE_GEN = env.pop('ARTEMIS_QED_TABLE_GEN', 'OFF')
ARTEMIS_DIMS = env.pop('ARTEMIS_DIMS', '1;2;3;RZ')
BUILD_PARALLEL = env.pop('BUILD_PARALLEL', '2')
BUILD_SHARED_LIBS = env.pop('ARTEMIS_BUILD_SHARED_LIBS',
                                   'OFF')
#BUILD_TESTING = env.pop('ARTEMIS_BUILD_TESTING',
#                               'OFF')
#BUILD_EXAMPLES = env.pop('ARTEMIS_BUILD_EXAMPLES',
#                                'OFF')
# openPMD-api sub-control
HDF5_USE_STATIC_LIBRARIES = env.pop('HDF5_USE_STATIC_LIBRARIES', 'OFF')
ADIOS_USE_STATIC_LIBS = env.pop('ADIOS_USE_STATIC_LIBS', 'OFF')
# CMake dependency control (developers & package managers)
ARTEMIS_AMREX_SRC = env.pop('ARTEMIS_AMREX_SRC', '')
ARTEMIS_AMREX_REPO = env.pop('ARTEMIS_AMREX_REPO', '')
ARTEMIS_AMREX_BRANCH = env.pop('ARTEMIS_AMREX_BRANCH', '')
ARTEMIS_AMREX_INTERNAL = env.pop('ARTEMIS_AMREX_INTERNAL', 'ON')
ARTEMIS_OPENPMD_SRC = env.pop('ARTEMIS_OPENPMD_SRC', '')
ARTEMIS_OPENPMD_INTERNAL = env.pop('ARTEMIS_OPENPMD_INTERNAL', 'ON')
ARTEMIS_PICSAR_SRC = env.pop('ARTEMIS_PICSAR_SRC', '')
ARTEMIS_PICSAR_INTERNAL = env.pop('ARTEMIS_PICSAR_INTERNAL', 'ON')
ARTEMIS_CCACHE_PROGRAM = env.pop('ARTEMIS_CCACHE_PROGRAM', None)

for key in env.keys():
    if key.lower().startswith('artemis'):
        print(f"\nWARNING: Found environment variable '{key}', which is not a recognized Artemis option\n")


# https://cmake.org/cmake/help/v3.0/command/if.html
if ARTEMIS_MPI.upper() in ['1', 'ON', 'TRUE', 'YES']:
    ARTEMIS_MPI = "ON"
else:
    ARTEMIS_MPI = "OFF"

# Include embedded boundary functionality
if ARTEMIS_EB.upper() in ['1', 'ON', 'TRUE', 'YES']:
    ARTEMIS_EB = "ON"
else:
    ARTEMIS_EB = "OFF"


# for CMake
cxx_modules = []     # values: artemis_1d, artemis_2d, artemis_3d, artemis_rz
cmdclass = {}        # build extensions

# externally pre-built: pick up pre-built Artemis libraries
if PYARTEMIS_LIB_DIR:
    cmdclass=dict(build=CopyPreBuild)
# CMake: build Artemis libraries ourselves
else:
    cmdclass = dict(build_ext=CMakeBuild)
    for dim in [x.lower() for x in ARTEMIS_DIMS.split(';')]:
        name = dim if dim == "rz" else dim + "d"
        cxx_modules.append(CMakeExtension("artemis_" + name))

# Get the package requirements from the requirements.txt file
install_requires = []
with open('./requirements.txt') as f:
    install_requires = [line.strip('\n') for line in f.readlines()]
    if ARTEMIS_MPI == "ON":
        install_requires.append('mpi4py>=2.1.0')

# keyword reference:
#   https://packaging.python.org/guides/distributing-packages-using-setuptools
setup(
    name='pyartemis',
    # note PEP-440 syntax: x.y.zaN but x.y.z.devN
    version = '23.04',
    packages = ['pyartemis'],
    package_dir = {'pyartemis': 'Python/pyartemis'},
    author='Jean-Luc Vay, David P. Grote, Maxence Thévenet, Rémi Lehe, Andrew Myers, Weiqun Zhang, Axel Huebl, et al.',
    author_email='jlvay@lbl.gov, grote1@llnl.gov, maxence.thevenet@desy.de, rlehe@lbl.gov, atmyers@lbl.gov, WeiqunZhang@lbl.gov, axelhuebl@lbl.gov',
    maintainer='Axel Huebl, David P. Grote, Rémi Lehe', # wheel/pypi packages
    maintainer_email='axelhuebl@lbl.gov, grote1@llnl.gov, rlehe@lbl.gov',
    description='Artemis is an advanced electromagnetic Particle-In-Cell code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=('Artemis openscience mpi hpc research pic particle-in-cell '
              'plasma laser-plasma accelerator modeling simulation'),
    url='https://ecp-warpx.github.io',
    project_urls={
        'Documentation': 'https://warpx.readthedocs.io',
        'Doxygen': 'https://warpx.readthedocs.io/en/latest/_static/doxyhtml/index.html',
        #'Reference': 'https://doi.org/...', (Paper and/or Zenodo)
        'Source': 'https://github.com/ECP-WarpX/WarpX',
        'Tracker': 'https://github.com/ECP-WarpX/WarpX/issues',
    },
    # CMake: self-built as extension module
    ext_modules=cxx_modules,
    cmdclass=cmdclass,
    # scripts=['artemis_1d', 'artemis_2d', 'artemis_3d', 'artemis_rz'],
    zip_safe=False,
    python_requires='>=3.7',
    # tests_require=['pytest'],
    install_requires=install_requires,
    # see: src/bindings/python/cli
    #entry_points={
    #    'console_scripts': [
    #        'artemis_3d = artemis.3d.__main__:main'
    #    ]
    #},
    extras_require={
        'all': ['openPMD-api~=0.15.1', 'openPMD-viewer~=1.1', 'yt>=4.1.0', 'matplotlib'],
    },
    # cmdclass={'test': PyTest},
    # platforms='any',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        ('License :: OSI Approved :: '
         'BSD License'), # TODO: use real SPDX: BSD-3-Clause-LBNL
    ],
    # new PEP 639 format
    license='BSD-3-Clause-LBNL',
    license_files = ['LICENSE.txt', 'LEGAL.txt'],
)
