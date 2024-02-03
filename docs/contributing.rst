Contributing
============

Pull requests are always welcomed. To make reviews fast and easy, before opening a PR,
please:

* Make sure any changes you're adding address general problems rather than niche
  problems specific to your use-case. ``ionics_fits`` aims to cover as many use-cases
  as possible, but there will always be some things which are too specific to one
  situation to make sense to include here. If in doubt, open an issue and ask!
* Make sure all changes are covered by a test (see :ref:`testing`).
* Check formatting: ``poe fmt``
* Run lints: ``poe flake``
* Check type annotations (Linux only): ``poe types``
* Run test suite: ``poe test``
* Update the documentation and :ref:`changes`
* Check the documentation builds: ``poe docs``
* Optionally, fuzz any new models: ``poe fuzz``