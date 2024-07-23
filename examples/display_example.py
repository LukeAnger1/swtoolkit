from swtoolkit.api.solidworks import SolidWorks

sw = SolidWorks()
results = sw.open("..\\tests\\test_cad\\test_native_assembly.SLDASM")

print(f'the results of opening are {results}')

model = sw.get_model()
# TODO: The below does not work, there is not enough parameters in the function
# model.set_configinfo("Engineered By:", "Tony Stark")

print(f'the model title is {model.title}')

# NOTE: This may have to be save, python wouldnt run with save but save3 runs so will have to look at this persons code
model.save3()