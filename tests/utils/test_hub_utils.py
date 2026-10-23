# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Cavaleiro AI (2026): add offline tests for repo ids and Hub paths derived from local paths.
import json
import os
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import constants, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled

from transformers import BertConfig, GenerationConfig, Wav2Vec2FeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, cached_file, has_file, list_repo_templates
from transformers.utils.hub import PushToHubMixin


RANDOM_BERT = "hf-internal-testing/tiny-random-bert"
TINY_BERT_PT_ONLY = "hf-internal-testing/tiny-bert-pt-only"
CACHE_DIR = os.path.join(constants.HF_HUB_CACHE, "models--hf-internal-testing--tiny-random-bert")
FULL_COMMIT_HASH = "9b8c223d42b2188cb49d29af482996f9d0f3e5a6"

GATED_REPO = "hf-internal-testing/dummy-gated-model"
README_FILE = "README.md"


class GetFromCacheTests(unittest.TestCase):
    def test_cached_file(self):
        archive_file = cached_file(RANDOM_BERT, CONFIG_NAME)
        # Should have downloaded the file in here
        self.assertTrue(os.path.isdir(CACHE_DIR))
        # Cache should contain at least those three subfolders:
        for subfolder in ["blobs", "refs", "snapshots"]:
            self.assertTrue(os.path.isdir(os.path.join(CACHE_DIR, subfolder)))
        with open(os.path.join(CACHE_DIR, "refs", "main"), encoding="utf-8") as f:
            main_commit = f.read()
        self.assertEqual(archive_file, os.path.join(CACHE_DIR, "snapshots", main_commit, CONFIG_NAME))
        self.assertTrue(os.path.isfile(archive_file))

        # File is cached at the same place the second time.
        new_archive_file = cached_file(RANDOM_BERT, CONFIG_NAME)
        self.assertEqual(archive_file, new_archive_file)

        # Using a specific revision to test the full commit hash.
        archive_file = cached_file(RANDOM_BERT, CONFIG_NAME, revision="9b8c223")
        self.assertEqual(archive_file, os.path.join(CACHE_DIR, "snapshots", FULL_COMMIT_HASH, CONFIG_NAME))

    def test_cached_file_errors(self):
        with self.assertRaisesRegex(EnvironmentError, "is not a valid model identifier"):
            _ = cached_file("tiny-random-bert", CONFIG_NAME)

        with self.assertRaisesRegex(EnvironmentError, "is not a valid git identifier"):
            _ = cached_file(RANDOM_BERT, CONFIG_NAME, revision="aaaa")

        with self.assertRaisesRegex(EnvironmentError, "does not appear to have a file named"):
            _ = cached_file(RANDOM_BERT, "conf")

    def test_non_existence_is_cached(self):
        with self.assertRaisesRegex(EnvironmentError, "does not appear to have a file named"):
            _ = cached_file(RANDOM_BERT, "conf")

        with open(os.path.join(CACHE_DIR, "refs", "main"), encoding="utf-8") as f:
            main_commit = f.read()
        self.assertTrue(os.path.isfile(os.path.join(CACHE_DIR, ".no_exist", main_commit, "conf")))

        path = cached_file(RANDOM_BERT, "conf", _raise_exceptions_for_missing_entries=False)
        self.assertIsNone(path)

        path = cached_file(RANDOM_BERT, "conf", local_files_only=True, _raise_exceptions_for_missing_entries=False)
        self.assertIsNone(path)

        # Under the mock environment, hf_hub_download will always raise an HTTPError
        with mock.patch(
            "transformers.utils.hub.hf_hub_download",
            side_effect=HfHubHTTPError("failed", response=mock.Mock(status_code=404)),
        ) as mock_head:
            path = cached_file(RANDOM_BERT, "conf", _raise_exceptions_for_connection_errors=False)
            self.assertIsNone(path)
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_has_file(self):
        self.assertTrue(has_file(TINY_BERT_PT_ONLY, WEIGHTS_NAME))
        self.assertFalse(has_file(TINY_BERT_PT_ONLY, "tf_model.h5"))
        self.assertFalse(has_file(TINY_BERT_PT_ONLY, "flax_model.msgpack"))

    def test_has_file_in_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Empty cache dir + offline mode => return False
            assert not has_file(TINY_BERT_PT_ONLY, WEIGHTS_NAME, local_files_only=True, cache_dir=tmp_dir)

            # Populate cache dir
            # TODO: only necessary for read-only cache systems; replace with a shared helper
            with unittest.mock.patch.dict(os.environ, {"HF_XET_CACHE": tmp_dir}):
                hf_hub_download(TINY_BERT_PT_ONLY, WEIGHTS_NAME, cache_dir=tmp_dir)

            # Cache dir + offline mode => return True
            assert has_file(TINY_BERT_PT_ONLY, WEIGHTS_NAME, local_files_only=True, cache_dir=tmp_dir)

    def test_get_file_from_repo_distant(self):
        # should return None if the file does not exist
        self.assertIsNone(
            cached_file(
                "google-bert/bert-base-cased",
                "ahah.txt",
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        )

        # The function raises if the repository does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid model identifier"):
            cached_file(
                "bert-base-case",
                CONFIG_NAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )

        # The function raises if the revision does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid git identifier"):
            cached_file(
                "google-bert/bert-base-cased",
                CONFIG_NAME,
                revision="ahaha",
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )

        resolved_file = cached_file(
            "google-bert/bert-base-cased",
            CONFIG_NAME,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        # The name is the cached name which is not very easy to test, so instead we load the content.
        config = json.loads(open(resolved_file, encoding="utf-8").read())
        self.assertEqual(config["hidden_size"], 768)

    def test_get_file_from_repo_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "a.txt"
            filename.touch()
            self.assertEqual(
                cached_file(
                    tmp_dir,
                    "a.txt",
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                ),
                str(filename),
            )

            self.assertIsNone(
                cached_file(
                    tmp_dir,
                    "b.txt",
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
            )

    def test_get_file_gated_repo(self):
        """Test download file from a gated repo fails with correct message when not authenticated."""
        with self.assertRaisesRegex(EnvironmentError, "You are trying to access a gated repo."):
            # All files except README.md are protected on a gated repo.
            cached_file(GATED_REPO, "gated_file.txt", token=False)

    def test_has_file_gated_repo(self):
        """Test check file existence from a gated repo fails with correct message when not authenticated."""
        with self.assertRaisesRegex(EnvironmentError, "is a gated repository"):
            # All files except README.md are protected on a gated repo.
            has_file(GATED_REPO, "gated_file.txt", token=False)

    def test_cached_files_exception_raised(self):
        """Test that unhadled exceptions, e.g. ModuleNotFoundError, is properly re-raised by cached_files when hf_hub_download fails."""
        with mock.patch(
            "transformers.utils.hub.hf_hub_download", side_effect=ModuleNotFoundError("No module named 'MockModule'")
        ):
            with self.assertRaises(ModuleNotFoundError):
                # The error should be re-raised by cached_files, not caught in the exception handling block
                cached_file(RANDOM_BERT, "nonexistent.json")


class OfflineModeTests(unittest.TestCase):
    def test_list_repo_templates_w_offline(self):
        with mock.patch("transformers.utils.hub.HfApi.list_repo_tree", side_effect=OfflineModeIsEnabled()):
            with mock.patch(
                "transformers.utils.hub.HfApi.snapshot_download",
                side_effect=LocalEntryNotFoundError("no snapshot found"),
            ):
                self.assertEqual(list_repo_templates(RANDOM_BERT, local_files_only=False), [])


class PushToHubLocalPathTests(unittest.TestCase):
    """Repo ids and Hub paths derived from local paths, checked without contacting the Hub."""

    def _assert_default_repo_id(self, obj, module):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dirs = (
                Path(tmp_dir) / "my-model",  # `os.PathLike`, as documented
                os.path.join(tmp_dir, "my-model") + os.sep,  # trailing separator
                f"{Path(tmp_dir).as_posix()}/my-model",  # forward slashes, also valid on Windows
            )
            for save_dir in save_dirs:
                with (
                    self.subTest(save_dir=save_dir),
                    mock.patch(f"{module}.hf_api") as mocked_hf_api,
                    mock.patch.object(type(obj), "_upload_modified_files"),
                ):
                    create_repo = mocked_hf_api.return_value.create_repo
                    create_repo.side_effect = lambda repo_id, **kwargs: mock.Mock(repo_id=repo_id)
                    obj.save_pretrained(save_dir, push_to_hub=True)
                    self.assertEqual(create_repo.call_args.args[0], "my-model")

    def test_default_repo_id_config(self):
        self._assert_default_repo_id(BertConfig(), "transformers.configuration_utils")

    def test_default_repo_id_generation_config(self):
        self._assert_default_repo_id(GenerationConfig(), "transformers.generation.configuration_utils")

    def test_default_repo_id_feature_extractor(self):
        self._assert_default_repo_id(Wav2Vec2FeatureExtractor(), "transformers.feature_extraction_utils")

    def test_default_repo_id_tokenizer(self):
        from transformers import BertTokenizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, "vocab.txt")
            with open(vocab_file, "w", encoding="utf-8") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n")
            tokenizer = BertTokenizer(vocab_file)
        self._assert_default_repo_id(tokenizer, "transformers.tokenization_utils_base")

    @require_torch
    def test_default_repo_id_model(self):
        from transformers import BertModel

        config = BertConfig(
            vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=8
        )
        with mock.patch("transformers.modeling_utils.create_and_tag_model_card"):
            self._assert_default_repo_id(BertModel(config), "transformers.modeling_utils")

    def test_upload_modified_files_uses_posix_path_in_repo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(os.path.join(tmp_dir, "additional_chat_templates"))
            for path in ("config.json", os.path.join("additional_chat_templates", "tool_use.jinja")):
                with open(os.path.join(tmp_dir, path), "w", encoding="utf-8") as f:
                    f.write("{}")
            with mock.patch("transformers.utils.hub.hf_api") as mocked_hf_api:
                PushToHubMixin()._upload_modified_files(tmp_dir, "user/my-model", files_timestamps={})
            operations = mocked_hf_api.return_value.create_commit.call_args.kwargs["operations"]
            self.assertEqual(
                sorted(op.path_in_repo for op in operations),
                ["additional_chat_templates/tool_use.jinja", "config.json"],
            )
