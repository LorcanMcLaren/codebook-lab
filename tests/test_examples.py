from __future__ import annotations

from pathlib import Path

import pytest

from codebook_lab.examples import (
    copy_example_task,
    get_example_task_dir,
    get_example_task_files,
    list_example_tasks,
)


class TestListExampleTasks:
    def test_contains_policy_sentiment(self):
        tasks = list_example_tasks()
        assert "policy-sentiment" in tasks

    def test_returns_sorted_list(self):
        tasks = list_example_tasks()
        assert tasks == sorted(tasks)


class TestGetExampleTaskDir:
    def test_valid_task(self):
        task_dir = get_example_task_dir("policy-sentiment")
        assert task_dir.exists()
        assert task_dir.is_dir()

    def test_invalid_task_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            get_example_task_dir("nonexistent-task")


class TestGetExampleTaskFiles:
    def test_returns_expected_keys(self):
        files = get_example_task_files("policy-sentiment")
        assert "task_dir" in files
        assert "codebook_path" in files
        assert "ground_truth_csv" in files

    def test_files_exist(self):
        files = get_example_task_files("policy-sentiment")
        assert files["codebook_path"].exists()
        assert files["ground_truth_csv"].exists()


class TestCopyExampleTask:
    def test_copy(self, tmp_path):
        result = copy_example_task("policy-sentiment", tmp_path)
        assert result.exists()
        assert (result / "codebook.json").exists()
        assert (result / "ground-truth.csv").exists()

    def test_copy_already_exists_raises(self, tmp_path):
        copy_example_task("policy-sentiment", tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            copy_example_task("policy-sentiment", tmp_path)

    def test_copy_overwrite(self, tmp_path):
        copy_example_task("policy-sentiment", tmp_path)
        result = copy_example_task("policy-sentiment", tmp_path, overwrite=True)
        assert result.exists()
