"""PyInstaller entry point for MediaIndex.exe."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mediaindex.desktop import main

    main()
