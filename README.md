# EYETALK2U

## C++ Configuration Guide
### OpenCV Setup
1. Download Prebuilt OpenCv from [This Link](https://opencv.org/releases/) and extract for example in [cvdir]
2. Download Visual Studio Community from [This Link](https://visualstudio.microsoft.com/downloads/)
3. Clone this Repo and open /cpp/eyetracking/eyetracking.sln
4. Right-click on project &rarr; Properties
5. Under Configuration: All Configurations:
- C/C++ &rarr; General &rarr; Additional Include Directories:
```[cvdir]\opencv\build\include```
- Linker → General → Additional Library Directories:
```[cvdir]\opencv\build\x64\vc16\lib```
- Linker → Input → Additional Dependencies: Add
```opencv_world[vers].lib``` and ```opencv_world[vers]d.lib```  
For example if your OpenCV version is 4.12.0 then add
```opencv_world4120.lib``` and ```opencv_world4120d.lib```
6. Copy the `opencv_world4120.dll` and `opencv_world4120d.dll` AND `opencv_videoio_ffmpeg4120_64.dll` from
  ```[cvdir]\opencv\build\x64\vc16\bin``` into the same folder as eyetracking.sln

### Parallelization Setup
1. Right-click on project &rarr; Properties
2. Under C/C++ &rarr; Optimization change `Optimization` from `Disabled (/Od)` to `Maximum Optimization (Favor Speed) (/O2)`
3. Under C/C++ &rarr; Code Generation change `Floating Point Model` from `Precise (/fp:precise)` to `Fast (/fp:fast)`
4. Under C/C++ &rarr; Language set Open MP Support to `Yes (/openmp)`

### Namespace Setup
1. Right-click on project &rarr; Properties
2. Under C/C++ &rarr; General &rarr; Additional Include Directories:  
```$(ProjectDir)include```  
```$(ProjectDir)tracking```

### C++ Version Setup [Important!]  
1. Right-click on project &rarr; Properties
2. Under C/C++ &rarr; Language change C++ Language Standard into `ISO C++20 Standard (/std:c++20)`

### uWebSockets Setup [Important!]
1. Clone `git clone https://github.com/microsoft/vcpkg.git` into any folder (DON'T CLONE ON THE SAME PROJECT)
2. Move to that folder and build vcpkg by running  
```
cd vcpkg  
.\bootstrap-vcpkg.bat
```
3. Integrate with your Visual Studio using this command  
```.\vcpkg integrate install```
4. Install Zlib by running  
```.\vcpkg install zlib:x64-windows```  
5. Install uWebSockets by running
```.\vcpkg install uwebsockets:x64-windows```
6. Just to be safe you can run `.\vcpkg integrate install` once again.
7. Done!

## 👥 Team Introduction

### **Ketua Kelompok**
**Naufal Septio Fathurrahman**  
NIM: 22/502670/TK/54886  
Program Studi: **Teknologi Informasi**  
Email: naufalseptiofathurrahman@mail.ugm.ac.id  

---

### **Anggota 1**
**Annisa Nabila**  
NIM: 22/493835/TK/54127  
Program Studi: **Teknik Biomedis**  
Email: annisanabila@mail.ugm.ac.id  

---

### **Anggota 2**
**Awaliya Shabrina**  
NIM: 22/494095/TK/54174  
Program Studi: **Teknik Biomedis**  
Email: awaliyashabrina@mail.ugm.ac.id  

---

### **Anggota 3**
**Cornelia Zefanya**  
NIM: 22/499785/TK/54764  
Program Studi: **Teknik Elektro**  
Email: corneliazefanya@mail.ugm.ac.id  

---

### **Anggota 4**
**Rama Sulaiman Nurcahyo**  
NIM: 22/492727/TK/53940  
Program Studi: **Teknologi Informasi**  
Email: ramasulaimannurcahyo@mail.ugm.ac.id  

---

### **Dosen Pembimbing**
**Dr.Eng. Ir. Sunu Wibirama, M.Eng., IPM.**  
NIP/NIU: 198510262015041003  

---
Departemen Teknik Elektro dan Teknologi Informasi  
Fakultas Teknik, Universitas Gadjah Mada  
