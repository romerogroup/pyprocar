from pyprocar.utils.download_examples import download_test_data


def main():
    """Command-line interface for download_test_data function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download test data from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_examples.py --relpath data/examples/00-band_structure/non-spin-polarized
  python download_examples.py --relpath data/examples/00-band_structure --output-path ./data
        """,
    )

    parser.add_argument(
        "--relpath",
        required=True,
        help="Relative path to the files to download (e.g., data/examples/00-band_structure/non-spin-polarized)",
    )

    parser.add_argument(
        "--output-path",
        default=".",
        help="Output directory path (default: current directory)",
    )

    args = parser.parse_args()

    try:
        print(f"Downloading files from: {args.relpath}")
        print(f"Output directory: {args.output_path}")

        download_dir = download_test_data(
            relpath=args.relpath, output_path=args.output_path
        )

        print(f"✅ Download completed successfully!")
        print(f"Files downloaded to: {download_dir}")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
