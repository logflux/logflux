{
  description = "A flake for the logflux suite of auomatic log parsers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {

    devShell.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      packages = [
        (nixpkgs.legacyPackages.x86_64-linux.python3.withPackages (python-pkgs: [
          python-pkgs.numpy
          python-pkgs.scikit-learn
          python-pkgs.torch
        ]))
      ];
      PYTHONPATH = self + "/src";
    };
  };
}

