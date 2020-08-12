from os import path as os_path

from setuptools import setup

import fast_alignment

this_directory = os_path.abspath(os_path.dirname(__file__))


# 读取文件内容
def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding="utf-8") as f:
        long_description = f.read()
    return long_description


# 获取依赖
def read_requirements(filename):
    return [
        line.strip()
        for line in read_file(filename).splitlines()
        if not line.startswith("#")
    ]


setup(
    name="fast-alignment",
    version=fast_alignment.__version__,
    description="Vectorized face alignment",
    author="Elliott Zheng",
    author_email="admin@hypercube.top",
    url="https://github.com/elliottzheng/fast-alignment",
    license="MIT",
    keywords="fast-alignment pytorch",
    project_urls={
        "Documentation": "https://github.com/elliottzheng/fast-alignment",
        "Source": "https://github.com/elliottzheng/fast-alignment",
        "Tracker": "https://github.com/elliottzheng/fast-alignment/issues",
    },
    # long_description=read_file("README.md"),  # 读取的Readme文档内容
    # long_description_content_type="text/markdown",  # 指定包文档格式为markdown
    packages=["fast_alignment"],
    install_requires=["numpy", "torch", "torchvision"],
    # package_data={'face_detection': ['weights/*.pth']}
)
