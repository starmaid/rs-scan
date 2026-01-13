{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (pkgs.python3.withPackages (python-pkgs: [
      pkgs.python3Packages.pyrealsense2
      pkgs.python3Packages.opencv-python
    ]))
    librealsense-gui
  ];
}