import os
import subprocess
from collections import OrderedDict
from pathlib import Path
import glob
import pytest
from sybil import Sybil
from sybil.parsers.markdown.codeblock import CodeBlockParser
from sybil.parsers.markdown.lexers import (
    FencedCodeBlockLexer,
    DirectiveInHTMLCommentLexer,
)
import docker


COMMAND_PREFIX = "$ "
IGNORED_OUTPUT = "..."
DIR_NEW_ENV = "new-env"
DIR_WORKDIR = "workdir"
DIR_SKIP_NEXT = "skip-next"
DIRECTIVE = "directive"
DIR_CODE_BLOCK = "code-block"
BASH = "bash"
ARGUMENTS = "arguments"
LINE_END = "\\"
CWD = "cwd"
EXPECTED_FILES = "expected-files"

REPO_ROOT_FOLDER = os.path.join(os.path.dirname(__file__), "..")


def evaluate_bash_block(example, cwd):
    """Executes a command and compares it's output to the provided expected output.

    ```bash
    command
    ```
    """
    print(f"{example=}")
    lines = example.strip().split("\n")
    output = []
    output_i = -1
    previous_cmd_line = ""
    next_is_cmd_continuation = False
    for line in lines:
        print(f"{line=}")
        if line.startswith(COMMAND_PREFIX) or next_is_cmd_continuation:
            # this is a command
            command = previous_cmd_line + line.replace(COMMAND_PREFIX, "")
            if command.endswith(LINE_END):
                # this must be merged with the next line
                previous_cmd_line = command.replace(LINE_END, "")
                next_is_cmd_continuation = True
                continue
            next_is_cmd_continuation = False
            print(f"{command=}")
            previous_cmd_line = ""
            # output = (
            #     subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, cwd=cwd)
            #     .strip()
            #     .decode("ascii")
            # )
            output = ""
            print(f"{output=}")
            output = [x.strip() for x in output.split("\n")]
            output_i = 0
        else:
            # this is expected output
            expected_line = line.strip()
            if len(expected_line) == 0:
                continue
            if output_i >= len(output):
                # end of captured output
                output_i = -1
                continue
            if output_i == -1:
                continue
            actual_line = output[output_i]
            while len(actual_line) == 0:
                # skip empty lines
                output_i += 1
                if output_i >= len(output):
                    output_i = -1
                    continue
                actual_line = output[output_i]
            if IGNORED_OUTPUT in expected_line:
                # skip this line
                output_i += 1
                continue
            assert actual_line == expected_line
            output_i += 1


def collect_docs():
    """Search for *.md files."""
    assert os.path.exists(REPO_ROOT_FOLDER), f"Path must exist: {REPO_ROOT_FOLDER}"

    pattern = os.path.join(REPO_ROOT_FOLDER, "*.md")
    all_md_files = glob.glob(pattern)
    print(f"Found {len(all_md_files)} .md files by {pattern}.")

    # Configure Sybil
    fenced_lexer = FencedCodeBlockLexer(BASH)
    new_env_lexer = DirectiveInHTMLCommentLexer(directive=DIR_NEW_ENV)
    workdir_lexer = DirectiveInHTMLCommentLexer(directive=DIR_WORKDIR)
    skip_lexer = DirectiveInHTMLCommentLexer(directive=DIR_SKIP_NEXT)
    sybil = Sybil(
        parsers=[fenced_lexer, new_env_lexer, workdir_lexer, skip_lexer],
        filenames=all_md_files,
    )
    documents = []
    for f_path in all_md_files:
        doc = sybil.parse(Path(f_path))
        rel_path = os.path.relpath(f_path, REPO_ROOT_FOLDER)
        if len(list(doc)) > 0:
            documents.append([rel_path, list(doc)])
    print(f"Found {len(documents)} .md files with code to test.")
    return documents


@pytest.mark.parametrize("path, blocks", collect_docs())
def test_doc_md(path, blocks):
    """Testing all code blocks in one *.md file under `path`."""
    print(f"Testing {len(blocks)} code blocks in {path}.")
    env_blocks = OrderedDict()
    env_workdirs = OrderedDict()
    env_arguments = OrderedDict()
    current_env = "ENV DEFAULT"
    workdir = None
    skip_next_block = False
    for block in blocks:
        if DIRECTIVE in block.region.lexemes:
            directive = block.region.lexemes[DIRECTIVE]
            if directive == DIR_NEW_ENV:
                arguments = block.region.lexemes["arguments"]
                assert arguments not in env_blocks.keys()
                current_env = f"ENV {block.path}:{block.line}"
                if ARGUMENTS in block.region.lexemes:
                    env_arguments[current_env] = block.region.lexemes[ARGUMENTS]
            elif directive == DIR_WORKDIR:
                workdir = block.region.lexemes[ARGUMENTS]
            elif directive == DIR_SKIP_NEXT:
                skip_next_block = True
            else:
                raise RuntimeError(f"Unsupported directive {directive}.")
        else:
            if skip_next_block:
                skip_next_block = False
                continue
            language = block.region.lexemes["language"]
            source = block.region.lexemes["source"]
            assert language == BASH, f"Unsupported language {language}"
            if current_env not in env_blocks.keys():
                env_blocks[current_env] = []
                env_workdirs[current_env] = []
            env_blocks[current_env].append(source)
            env_workdirs[current_env].append(workdir)
            workdir = None
    # After preprocessing all the environments, evaluate them.
    for env, blocks in env_blocks.items():
        # Get arguments
        assert env in env_arguments, "Environment must have arguments."
        arguments = env_arguments[env]
        print(f"Evaluating environment >{env}< with {arguments=} ...")
        client = docker.from_env()
        arguments = arguments.split(" ")
        assert len(arguments) == 1, f"Expecting one argument. Got {arguments}"
        image = arguments[0]
        client.images.pull(image)
        container = client.containers.run(
            image=image, command="sleep 1d", auto_remove=True, detach=True
        )
        # Get workdirs
        assert env in env_workdirs, "Environment must have working directories."
        workdirs = env_workdirs[env]
        assert len(blocks) == len(workdirs), "Must have one workdir entry per block."
        try:
            for block, workdir in zip(blocks, workdirs):
                for line in block.split("\n"):
                    command = f'bash -ic "{line.strip()}"'
                    if len(line) == 0:
                        continue
                    print(f"Executing {command=}")
                    returncode, output = container.exec_run(
                        command, tty=True, workdir=workdir
                    )
                    output = output.decode("utf-8")
                    if returncode == 0:
                        print("✅")
                        print("output ...")
                        print(output)
                    else:
                        print(f"❌ {returncode=}")
                    assert (
                        returncode == 0
                    ), f"Unexpected Returncode: {returncode=}\noutput ...\n{output}"
        except Exception as e:
            print(e)
            assert False, str(e)
        finally:
            container.stop()
