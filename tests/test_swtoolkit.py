from swtoolkit import __version__


def test_version():
    print(f'the version is {__version__}')
    assert __version__ == '0.1.0'

if __name__ == '__main__':
    # testing version
    test_version()
