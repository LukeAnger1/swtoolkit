import subprocess

import win32gui
import win32process
import pytest
import psutil
import os


from swtoolkit import SolidWorks
from swtoolkit.api.com import COM


def visibility_state():
    """Gets the visibility state of SLDWORKS.exe via 3rd party package

    Returns True if window is visible; False if not. This function assumes that
    if the solidworks process is in the process list but does not return a
    win32 window handle based on the callback() logic then it must be
    invisible.
    """

    target_pid = [p.pid for p in psutil.process_iter() if p.name() == "SLDWORKS.exe"]

    def callback(handle, handles):
        if win32gui.IsWindowEnabled(handle) and win32gui.IsWindowVisible(handle):
            _, pid = win32process.GetWindowThreadProcessId(handle)
            if pid in target_pid:
                handles.append(handle)
        return True

    handles = []
    win32gui.EnumWindows(callback, handles)

    handle_list = [handle for handle in handles if bool(win32gui.GetWindowText(handle))]

    # Removes the error message box that SolidWorks triggers on Startup from
    # the list of handles used to determine if solidworks is visible
    MESSAGE_BOX_CLASS = 32770
    patch_handle = win32gui.FindWindow(
        MESSAGE_BOX_CLASS, "Microsoft Visual Basic for Applications"
    )

    if patch_handle in handle_list:
        handle_list.remove(patch_handle)

    return bool(handle_list)


def test_start():
    """Checks if SolidWorks is in the process list before and after
    execution of start().
    """
    SolidWorks.start()
    assert "SLDWORKS.exe" in [p.name() for p in psutil.process_iter()]


@pytest.mark.usefixtures("teardown_sldworks")
def test_kill():
    """Checks if SolidWorks is not in the process list before and after
    execution of kill().
    """

    if "SLDWORKS.exe" not in [p.name() for p in psutil.process_iter()]:
        SolidWorks.start()
    SolidWorks.kill()
    # print(f'the running files are {[p.name() for p in psutil.process_iter()]}')
    assert "SLDWORKS.exe" not in [p.name() for p in psutil.process_iter()]


def test_pid():
    """Check if PID returned by SolidWorks matches PID in process list"""

    SolidWorks.start()
    assert SolidWorks().pid in [
        p.pid for p in psutil.process_iter() if p.name() == "SLDWORKS.exe"
    ]


def test_visible_get():
    """Check if SolidWorks returns its visibility state. Default state is
    False
    """
    assert SolidWorks().visible is False


def test_visible_set():
    """Check if visibility state can be set via changing attribute bool
    value
    """
    SolidWorks().visible = True
    assert SolidWorks().visible is True


@pytest.mark.parametrize(
    "filename",
    [
        os.path.join(os.getcwd(), filename)
        for filename in [
            "tests/test_cad/test_native_part.SLDPRT",
            "tests/test_cad/test_native_assembly.SLDASM",
            "tests/test_cad/test_native_drawing.SLDDRW",
        ]
    ],
)
def test_open(filename):
    """ Test if native solidworks files can be opened """

    retval = SolidWorks().open(filename)
    assert retval[0]  # Returned value is true if file is opened successfully.
    SolidWorks().close_all_documents() # TODO: issue with parameters here


def test_shutdown():
    pass


def test_get_model():
    pass


def test_get_models():
    pass

def test_rotating_CAD():
    # NOTE: See the below for more details
    # https://github.com/onuralpszr/SWPyMacros/blob/master/hello_python_world.py

    import win32com
    import pythoncom
    import time
    SWV = 2023 # This is solidworks version year
    SWAV = SWV-1992 # API Version
    
    sw = win32com.client.Dispatch("SldWorks.Application.{}".format(SWAV)) 
    print(f" Solidworks API Version : {SWAV}","\n",f"Solidworks Version : {SWV}")
    Model = sw.ActiveDoc
    ARG_NULL = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)
    ck = Model.Extension.SelectByID2("Front", "PLANE", 0, 0, 0, False, 0, ARG_NULL, 0)
    Model.SketchManager.InsertSketch(True)
    Model.ClearSelection2(True)

    mySketchText = Model.InsertSketchText(
        0, 0, 0, "Hello Python World!", 0, 0, 0, 100, 100
    )
    myFeature = Model.FeatureManager.FeatureExtrusion2(
        True, 
        False,
        False,
        0,
        0,
        0.001,
        0.001,
        False,
        False,
        False,
        False,
        0,
        0,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        0,
        0,
        False,
    )

    Model.SelectionManager.EnableContourSelection = False
    Model.ClearSelection2(True)
    s = time.perf_counter()
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds until while-loop")
    time.sleep(2)
    while True:
        Model.ViewRotateplusy()
        time.sleep(0.1)

if __name__ == '__main__':
    # Testing visibiliity
    # print(f'the visibility is {visibility_state()}')
    # test_start()
    test_rotating_CAD()
    test_open('woodAndScrewCounter/testing/swtoolkit/tests/test_cad/test_asm/TEST-AAA-AA.SLDASM')
    # test kill
    # test_kill()
    # test pid
    # test_pid()
    # test the visibility
    # test_visible_get()
    # Test set visibility
    # test_visible_set()
