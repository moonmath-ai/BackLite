# ncu -set full -f back_lite%i.ncu-rep ./test_back_lite.py

ncu \
  --set full \
  -o back_lite%i \
  --kernel-name device_kernel \
  python profile_back_lite.py