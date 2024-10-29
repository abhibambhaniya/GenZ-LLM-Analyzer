import argparse
import copy
import importlib
import os

try:
    # before refactory folders of chakra
    importlib.import_module("chakra.et_def.et_def_pb2")
    from chakra.et_def.et_def_pb2 import GlobalMetadata
    from chakra.et_def.et_def_pb2 import Node
    from chakra.et_def.et_def_pb2 import NodeType
    from chakra.et_def.et_def_pb2 import AttributeProto as ChakraAttr
    from chakra.third_party.utils.protolib import encodeMessage as encode_message
    from chakra.third_party.utils.protolib import decodeMessage as decode_message
    from chakra.third_party.utils.protolib import openFileRd as open_file_rd
except ImportError:
    # after refactory
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata
    from chakra.schema.protobuf.et_def_pb2 import Node
    from chakra.schema.protobuf.et_def_pb2 import NodeType
    from chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
    from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
    from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
    from chakra.src.third_party.utils.protolib import openFileRd as open_file_rd


required_type = {
    "num_ops": "int64_val",
    "tensor_size": "uint64_val",
    "comm_priority": "int32_val",
    "comm_size": "int64_val",
    "comm_src": "int32_val",
    "comm_dst": "int32_val",
    "comm_tag": "int32_val"
}


def process_attr(attr):
    name = attr.name
    if name in required_type:
        new_type = required_type[name]
    else:
        return attr
    ret_attr = ChakraAttr(name=attr.name)
    original_type = attr.WhichOneof("value")
    value = getattr(attr, original_type)
    setattr(ret_attr, new_type, value)
    return ret_attr
    

def process_chakra_node(node):
    attrs = copy.deepcopy(node.attr)
    for _ in range(len(node.attr)):
        node.attr.pop()
    assert len(node.attr) == 0
    for attr in attrs:
        new_attr = process_attr(attr)
        node.attr.append(new_attr)
    if node.type in {NodeType.COMM_COLL_NODE, NodeType.COMM_SEND_NODE, NodeType.COMM_RECV_NODE}:
        node.attr.append(
            ChakraAttr(name="is_cpu_op", bool_val=False)
        )
    return node


def convert_chakra_file(input_filename, output_filename):
    fr = open_file_rd(input_filename)
    fw = open(output_filename, "wb")
    gm = GlobalMetadata()
    node = Node()
    decode_message(fr, gm)
    encode_message(fw, gm)
    while decode_message(fr, node):
        node = process_chakra_node(node)
        encode_message(fw, node)
    fr.close()
    fw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="fix type issue of chakra attrs, either specify a single file or a folder contains lots of files."
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default=None,
        required=False,
        help="input chakra filename"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        required=False,
        help="output chakra filename"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        required=False,
        help="folder contains input chakra files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        required=False,
        help="folder contains input chakra files"
    )
    args = parser.parse_args()
    
    folder_mod = (args.input_folder is not None)
    if folder_mod:
        if (args.input_folder is None) or (args.output_folder is None):
            parser.print_help()
            exit(1)
        os.makedirs(args.output_folder, exist_ok=True)
        files = os.listdir(args.input_folder)
        for file in files:
            if not file.endswith(".et"):
                continue
            input_file = os.path.join(args.input_folder, file)
            output_file = os.path.join(args.output_folder, file)
            convert_chakra_file(input_file, output_file)
    else:
        if (args.input_filename is None) or (args.output_filename is None):
            parser.print_help()
            exit(1)
        convert_chakra_file(args.input_filename, args.output_filename)
    