"""PyInstaller entry point for MediaIndex.app."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mediaindex.desktop import main

    main()
