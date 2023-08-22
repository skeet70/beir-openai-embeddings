{
  description = "Devshell for downloading new embedding datasets.";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "github:DavHau/mach-nix/3.5.0";
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = mach-nix.lib.${system}.mkPython
          {
            requirements = ''
              black
              requests
              tqdm
            '';
          };
      in
      {
        devShell = pkgs.mkShell {
          nativeBuildInputs = [ ];
          buildInputs = [
            pythonEnv
            pkgs.ruff
            pkgs.git-lfs
          ];
        };
      });
}
