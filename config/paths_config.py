from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class PathsConfig:
    """
    File system locations used by the app

    All paths are stored as Path objects and normalized (expanded + resolved).
    Output directories can be created with `ensure_output_dirs()`
    """
    input_docx_folder: Path
    output_docx_folder: Path
    explained_txt_folder: Path

    def list_input_docx(self) -> list[Path]:
        return sorted(self.input_docx_folder.glob("*.docx"))

    @staticmethod
    def from_strings(
        input_docx_folder: str | Path,
        output_docx_folder: str | Path,
        explained_txt_folder: str | Path,
    ) -> "PathsConfig":
        """
        Convenience constructor for CLI/env usage.
        
        :param input_docx_folder: Description
        :type input_docx_folder: str | Path
        :param output_docx_folder: Description
        :type output_docx_folder: str | Path
        :param explained_txt_folder: Description
        :type explained_txt_folder: str | Path
        :return: Description
        :rtype: PathsConfig
        """
        return PathsConfig(
            input_docx_folder=PathsConfig._norm(input_docx_folder),
            output_docx_folder=PathsConfig._norm(output_docx_folder),
            explained_txt_folder=PathsConfig._norm(explained_txt_folder),
        )
    
    def ensure_output_dirs(self) -> None:
        """
        Create output directories if they don't exist.
        """
        self.output_docx_folder.mkdir(parents=True, exist_ok=True)
        self.explained_txt_folder.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """
       Validate that inputs exist and are directories.
       Raises ValueError with a helpful message if something is wrong
        """
        if not self.input_docx_folder.exists():
            raise ValueError(f"Input folder does not exist: {self.input_docx_folder}")
        if not self.input_docx_folder.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_docx_folder}")
        
        # Outputs can be created; but if they exist and aren't dirs, that's an error
        for p, label in [
            (self.output_docx_folder, "output_docx_folder"),
            (self.explained_txt_folder, "explained_txt_folder"),
        ]:
            if p.exists() and not p.is_dir():
                raise ValueError(f"{label} exists but is not a directory: {p}")
            
    @staticmethod
    def _norm(p: str | Path) -> Path:
        """
        Normalize a path: expand ~ and resolve to an absolute path.
        """
        return Path(p).expanduser().resolve()
