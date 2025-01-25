
### Example Data Folder Structure

Organize your data folder as follows:

- **`audio/`**: Contains the audio files.
- **`metadata/`**: Contains CSV files with metadata. Each CSV file must include the following columns:
  - **`file`**: File path to the corresponding audio file.
  - **`label`**: String label associated with the audio file (e.g., `bonafide` or `spoof`).

---

### Example Preprocessing Script

To create metadata files, use the preprocessing script provided:

```bash
python data_process/create_meta.py