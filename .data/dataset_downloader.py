
import os
import shutil
import argparse
import posixpath
from pathlib import Path
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.parse import urljoin, urlsplit, unquote, urldefrag


class LinkParser(HTMLParser):
    """Extract all hyperlinks from an HTML directory listing."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for attr, value in attrs:
                if attr.lower() == "href" and value:
                    self.links.append(value)


def get_directory_links(directory_url):
    """Retrieve and parse links from a directory listing."""

    request = Request(
        directory_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = LinkParser()
    parser.feed(html)

    return parser.links


def is_inside_root(url, root_url):
    """Ensure the URL belongs to the original dataset directory."""

    parsed_url = urlsplit(url)
    parsed_root = urlsplit(root_url)

    # Only allow the same server and HTTPS/HTTP scheme.
    if parsed_url.netloc.lower() != parsed_root.netloc.lower():
        return False

    if parsed_url.scheme.lower() != parsed_root.scheme.lower():
        return False

    root_path = posixpath.normpath(
        unquote(parsed_root.path)
    )

    target_path = posixpath.normpath(
        unquote(parsed_url.path)
    )

    return (
        target_path == root_path
        or target_path.startswith(root_path.rstrip("/") + "/")
    )


def get_relative_path(url, root_url):
    """Get the complete relative path from the dataset root."""

    target_path = posixpath.normpath(
        unquote(urlsplit(url).path)
    )

    root_path = posixpath.normpath(
        unquote(urlsplit(root_url).path)
    )

    relative_path = posixpath.relpath(
        target_path,
        root_path
    )

    if relative_path == ".":
        return None

    parts = relative_path.split("/")

    # Prevent unsafe filesystem paths.
    if any(part in ("", ".", "..") for part in parts):
        return None

    if any("\\" in part or ":" in part for part in parts):
        return None

    return Path(*parts)


def download_file(file_url, destination):
    """Download a file safely using a temporary file."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists():
        print(f"Skipping (already exists): {destination}")
        return

    request = Request(
        file_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    temp_file = destination.with_name(
        destination.name + ".part"
    )

    try:
        print(f"Downloading: {destination}")

        with urlopen(request, timeout=120) as response:
            with open(temp_file, "wb") as output:
                shutil.copyfileobj(response, output)

        # Rename only after the download completes successfully.
        os.replace(temp_file, destination)

        print(f"Completed: {destination}")

    except Exception as error:
        print(f"FAILED: {file_url}")
        print(f"Error: {error}")

        if temp_file.exists():
            temp_file.unlink()


def download_directory(root_url, output_dir):
    """
    Recursively download every accessible file and subdirectory.

    Preserves the complete original directory hierarchy.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store directories waiting to be processed.
    pending_directories = [root_url]

    # Prevent visiting the same directory multiple times.
    visited_directories = set()

    total_files = 0
    failed_directories = 0

    while pending_directories:

        current_url = pending_directories.pop()

        current_url = current_url.rstrip("/") + "/"

        if current_url in visited_directories:
            continue

        visited_directories.add(current_url)

        print("\n" + "=" * 70)
        print(f"Entering directory: {current_url}")
        print("=" * 70)

        try:
            links = get_directory_links(current_url)

        except Exception as error:
            print(f"Failed to access directory: {current_url}")
            print(f"Error: {error}")
            failed_directories += 1
            continue

        for link in links:

            # Ignore empty links and unsupported schemes.
            if (
                not link
                or link.startswith("?")
                or link.startswith("#")
                or link.startswith("javascript:")
                or link.startswith("mailto:")
            ):
                continue

            # Resolve relative URLs.
            full_url = urljoin(current_url, link)

            # Remove fragments.
            full_url, _ = urldefrag(full_url)

            parsed_url = urlsplit(full_url)

            # Ignore directory-listing sorting parameters.
            if parsed_url.query:
                continue

            # Prevent crawling outside the dataset root.
            if not is_inside_root(full_url, root_url):
                continue

            # Construct the complete local relative path.
            relative_path = get_relative_path(
                full_url,
                root_url
            )

            if relative_path is None:
                continue

            destination = output_dir / relative_path

            # Identify subdirectories.
            if parsed_url.path.endswith("/"):
                if full_url not in visited_directories:
                    print(f"Found directory: {relative_path}")

                    pending_directories.append(full_url)

                # Create the directory locally.
                destination.mkdir(
                    parents=True,
                    exist_ok=True
                )

            else:
                # Download files.
                download_file(
                    full_url,
                    destination
                )

                total_files += 1

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Directories processed: {len(visited_directories)}")
    print(f"Files attempted:       {total_files}")
    print(f"Failed directories:    {failed_directories}")
    print(f"Output directory:      {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Recursively download the complete NOAA IBTrACS "
            "dataset while preserving all directory levels."
        )
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v04r01",
        help="IBTrACS version (default: v04r01)"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="access/netcdf/",
        help="directory inside url"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory"
    )

    args = parser.parse_args()

    root_url = (
        "https://www.ncei.noaa.gov/data/"
        "international-best-track-archive-for-climate-stewardship-ibtracs/"
        f"{args.version}/{args.dir}"
    )

    output_dir = args.output or f"{args.version}/{args.dir}"

    print("\nNOAA IBTrACS Recursive Downloader")
    print("-" * 70)
    print(f"Dataset version: {args.version}")
    print(f"Source URL:      {root_url}")
    print(f"Output folder:   {os.path.abspath(output_dir)}")
    print("-" * 70)

    download_directory(
        root_url=root_url,
        output_dir=output_dir
    )

    print("\nAll accessible directory levels have been processed.")
