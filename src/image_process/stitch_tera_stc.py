import os
import subprocess
import xml.etree.ElementTree as ET


# def generate_terastc_xml(stacks_dir, voxel_dims, tile_height, tile_width, N_slices, X_tiles, Y_tiles, output_xml):
#     """
#     生成 TeraStitcher 的 XML 导入文件。

#     参数：
#     --------
#     stacks_dir : str
#         tiles 所在的目录路径。
#     voxel_dims : dict
#         体素尺寸，格式为 {"V": "1.0", "H": "1.0", "D": "1.0"}。
#     tile_height : str
#         tile 的高度（像素数）。
#     tile_width : str
#         tile 的宽度（像素数）。
#     N_slices : str
#         tile 在 Z 方向上的切片数。
#     X_tiles : int
#         X 方向的 tile 数目（列数）。
#     Y_tiles : int
#         Y 方向的 tile 数目（行数）。
#     output_xml : str
#         输出的 XML 文件名。
#     """

#     # 创建根节点
#     root = ET.Element("TeraStitcher", volume_format="TiledXY|3Dseries", input_plugin="tiff3D")

#     # 添加子节点
#     ET.SubElement(root, "stacks_dir", value=stacks_dir)
#     ET.SubElement(root, "ref_sys", ref1="X", ref2="Y", ref3="Z")
#     ET.SubElement(root, "voxel_dims", V=voxel_dims["V"], H=voxel_dims["H"], D=voxel_dims["D"])
#     ET.SubElement(root, "origin", V="0.0", H="0.0", D="0.0")
#     ET.SubElement(root, "mechanical_displacements", V="0.0", H="0.0", D="0.0")
#     ET.SubElement(root, "dimensions", stack_rows=str(Y_tiles), stack_columns=str(X_tiles), stack_slices=N_slices)

#     stacks = ET.SubElement(root, "STACKS")

#     # 计算 tile 总数
#     total_tiles = X_tiles * Y_tiles

#     # 创建 BLOCK 节点
#     tile_number = 1  # 从 1 开始编号
#     for y in range(Y_tiles):
#         for x in range(X_tiles):
#             # 计算实际的 tile 编号，按照从左到右，从下到上的顺序
#             tile_id = tile_number
#             tile_number += 1

#     # 重置 tile_number，按照指定的编号顺序
#     tile_number = 1
#     for y in range(Y_tiles - 1, -1, -1):
#         for x in range(X_tiles - 1, -1, -1):
#             block = ET.SubElement(stacks, "Stack", block_id=str(tile_number), z="0", y=str(y), x=str(x))
#             stack = ET.SubElement(block, "STACK", N_CHANS="1", N_BYTESxCHAN="2", ROWS=tile_height, COLS=tile_width)
#             ET.SubElement(stack, "ABS_V", value="0.0")
#             ET.SubElement(stack, "ABS_H", value="0.0")
#             ET.SubElement(stack, "ABS_D", value="0.0")
#             ET.SubElement(stack, "STITCHABLE", value="yes")
#             ET.SubElement(stack, "DIR_NAME", value="")
#             # 假设文件名为 "tile_编号.tif"
#             file_name = f"TestTile{tile_number:03d}.tif"
#             ET.SubElement(stack, "FILE_NAME", value=file_name)
#             tile_number += 1

#     # 生成 XML 字符串
#     tree = ET.ElementTree(root)
#     xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')

#     # 保存到文件
#     with open(output_xml, "w") as f:
#         f.write(xml_str)

def generate_terastc_xml(stacks_dir, voxel_dims, tile_height, tile_width, N_slices, X_tiles, Y_tiles, output_xml):
    """
    生成符合 TeraStitcher 规范的 XML 文件，用于 3D 图像拼接。

    参数：
    --------
    stacks_dir : str
        tiles 所在的目录路径。
    voxel_dims : dict
        体素尺寸，格式为 {"V": "1.0", "H": "1.0", "D": "1.0"}。
    tile_height : str
        tile 的高度（像素数）。
    tile_width : str
        tile 的宽度（像素数）。
    N_slices : str
        tile 在 Z 方向上的切片数。
    X_tiles : int
        X 方向的 tile 数目（列数）。
    Y_tiles : int
        Y 方向的 tile 数目（行数）。
    output_xml : str
        输出的 XML 文件名。
    """

    # 创建根节点
    root = ET.Element("TeraStitcher", volume_format="TiledXY|3Dseries", input_plugin="tiff3D")
    ET.SubElement(root, "stacks_dir", value=stacks_dir)
    ET.SubElement(root, "ref_sys", ref1="1", ref2="2", ref3="3")
    ET.SubElement(root, "voxel_dims", V=voxel_dims["V"], H=voxel_dims["H"], D=voxel_dims["D"])
    ET.SubElement(root, "origin", V="0", H="0", D="0")
    ET.SubElement(root, "mechanical_displacements", V="1000", H="1000")
    ET.SubElement(root, "dimensions", stack_rows=str(X_tiles), stack_columns=str(Y_tiles), stack_slices=N_slices)

    stacks = ET.SubElement(root, "STACKS")

    tile_number = 1
    for y in range(Y_tiles-1,-1,-1):
        for x in range(X_tiles-1,-1,-1):
            stack = ET.SubElement(stacks, "Stack", N_BLOCKS="1", BLOCK_SIZES=N_slices, BLOCKS_ABS_D="0", 
                                  N_CHANS="1", N_BYTESxCHAN="1", 
                                  ROW=str(y), 
                                  COL=str(x), 
                                  ABS_V=str(tile_height*y), ABS_H=str(tile_width*x), ABS_D="0", 
                                  STITCHABLE="no", 
                                  DIR_NAME=f"TestTile{tile_number:03d}", 
                                  Z_RANGES=f"[0,{N_slices})", 
                                  IMG_REGEX="")

            # 添加 NORTH, EAST, SOUTH, WEST displacements
            ET.SubElement(stack, "NORTH_displacements")
            ET.SubElement(stack, "EAST_displacements")
            ET.SubElement(stack, "SOUTH_displacements")
            ET.SubElement(stack, "WEST_displacements")

            tile_number += 1

    # 生成 XML 树并保存到文件
    tree = ET.ElementTree(root)
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)

def run_command(cmd):
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)  # 打印标准输出
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Error Output:\n{e.stderr}")

def terastitcher_workflow(xml_file, output_dir):
    # 步骤 1：导入
    import_cmd = [
        'terastitcher', '--import',
        '--projin=' + xml_file,
        '--volin_plugin=tiff3D'
    ]
    run_command(import_cmd)

    # 步骤 2：位移计算
    displcompute_xml = os.path.join(output_dir, 'displcompute.xml')
    displcompute_cmd = [
        'terastitcher', '--displcompute',
        '--projin=' + xml_file,
        '--projout=' + displcompute_xml,
        '--algorithm=MIPNCC',  # 使用 MIPNCC 算法
    ]
    run_command(displcompute_cmd)

    # 步骤 3：位移投影
    displproj_xml = os.path.join(output_dir, 'displproj.xml')
    displproj_cmd = [
        'terastitcher', '--displproj',
        '--projin=' + displcompute_xml,
        '--projout=' + displproj_xml
    ]
    run_command(displproj_cmd)
    
    # 步骤 4：阈值处理（可选）
    threshold_xml = os.path.join(output_dir, 'threshold.xml')
    threshold_cmd = [
        'terastitcher', '--threshold',
        '--projin=' + displproj_xml,
        '--projout=' + threshold_xml
    ]
    run_command(threshold_cmd)

    # 步骤 5：放置图块
    placetiles_xml = os.path.join(output_dir, 'placetiles.xml')
    placetiles_cmd = [
        'terastitcher', '--placetiles',
        '--projin=' + threshold_xml,
        '--projout=' + placetiles_xml
    ]
    run_command(placetiles_cmd)

    # 步骤 6：合并
    merge_cmd = [
        'terastitcher', '--merge',
        '--projin=' + placetiles_xml,
        '--volout=' + output_dir,
        '--volout_plugin=TIFF3D',
        '--imout_format=TIFF (uncompressed)',
        '--resolutions=0',
    ]
    run_command(merge_cmd)
