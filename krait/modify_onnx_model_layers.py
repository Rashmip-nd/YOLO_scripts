import onnx

model = onnx.load('/Users/rashmipulgam/ND_dev/snpe_krait_dev/snpe-1.52.0.2724/models/tycho_yolo/data/test_single_image/aimet_models/quantized_aimet_nd_yolo.onnx')

node_names_to_remove_transpose = ["/Transpose", "/Transpose_1", "/Transpose_2"]
node_names_to_remove_reshape = ["/Reshape", "/Reshape_1", "/Reshape_2"]
global node_count
nodes_to_remove = []

def remove_node_by_name(model, node_names_to_remove):
    node_count = 0
    for idx, node in enumerate(model.graph.node):
        for node_name_to_remove in node_names_to_remove:
            if node_name_to_remove == node.name:
                node_count += 1
                print("idx : ", idx)
                print(node.input)
                while node.input:
                    model.graph.node[idx].input.pop()

                while node.output:
                    model.graph.node[idx].output.pop()

                # for i in range(len(node.input)):
                #     model.graph.node[idx].input.remove(node.input[0])
                # model.graph.node[idx].output.remove(node.output[0])
                del model.graph.node[idx]
    print("Found ", node_count, " nodes")
    return model

model = remove_node_by_name(model, node_names_to_remove_transpose)
model = remove_node_by_name(model, node_names_to_remove_reshape)

#Rename the last layer output
for idx, node in enumerate(model.graph.node):
    if '/0/Conv_output_0' == node.output[0]:
        print(node.output[0])
        node.output[0] = "Conv_output_00"
        print(node.output[0])

    if '/1/Conv_output_0' == node.output[0]:
        print(node.output[0])
        node.output[0] = "Conv_output_01"
        print(node.output[0])
    
    if '/2/Conv_output_0' == node.output[0]:
        print(node.output[0])
        node.output[0] = "Conv_output_02"
        print(node.output[0])

#Modify the model output to new outputs
while model.graph.output:
    model.graph.output.pop()

model.graph.output.append(helper.make_tensor_value_info('Conv_output_00', TensorProto.FLOAT, [1,48,80,57]))
model.graph.output.append(helper.make_tensor_value_info('Conv_output_01', TensorProto.FLOAT, [1,24,40,57])) 
model.graph.output.append(helper.make_tensor_value_info('Conv_output_02', TensorProto.FLOAT, [1,12,20,57]))

# Save the modified model
onnx.save(model, '/Users/rashmipulgam/ND_dev/snpe_krait_dev/snpe-1.52.0.2724/models/tycho_yolo/data/test_single_image/aimet_models/modified_quantized_aimet_nd_yolo.onnx')

print("Last layer removed and new model saved as 'modified_quantized_aimet_nd_yolo.onnx'.")