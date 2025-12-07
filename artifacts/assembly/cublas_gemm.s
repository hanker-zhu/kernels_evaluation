
./cublas_gemm:     file format elf64-x86-64


Disassembly of section .init:

0000000000002000 <_init>:
    2000:	f3 0f 1e fa          	endbr64
    2004:	48 83 ec 08          	sub    $0x8,%rsp
    2008:	48 8b 05 d9 6f 00 00 	mov    0x6fd9(%rip),%rax        # 8fe8 <__gmon_start__@Base>
    200f:	48 85 c0             	test   %rax,%rax
    2012:	74 02                	je     2016 <_init+0x16>
    2014:	ff d0                	call   *%rax
    2016:	48 83 c4 08          	add    $0x8,%rsp
    201a:	c3                   	ret

Disassembly of section .plt:

0000000000002020 <.plt>:
    2020:	ff 35 02 6e 00 00    	push   0x6e02(%rip)        # 8e28 <_GLOBAL_OFFSET_TABLE_+0x8>
    2026:	ff 25 04 6e 00 00    	jmp    *0x6e04(%rip)        # 8e30 <_GLOBAL_OFFSET_TABLE_+0x10>
    202c:	0f 1f 40 00          	nopl   0x0(%rax)
    2030:	f3 0f 1e fa          	endbr64
    2034:	68 00 00 00 00       	push   $0x0
    2039:	e9 e2 ff ff ff       	jmp    2020 <_init+0x20>
    203e:	66 90                	xchg   %ax,%ax
    2040:	f3 0f 1e fa          	endbr64
    2044:	68 01 00 00 00       	push   $0x1
    2049:	e9 d2 ff ff ff       	jmp    2020 <_init+0x20>
    204e:	66 90                	xchg   %ax,%ax
    2050:	f3 0f 1e fa          	endbr64
    2054:	68 02 00 00 00       	push   $0x2
    2059:	e9 c2 ff ff ff       	jmp    2020 <_init+0x20>
    205e:	66 90                	xchg   %ax,%ax
    2060:	f3 0f 1e fa          	endbr64
    2064:	68 03 00 00 00       	push   $0x3
    2069:	e9 b2 ff ff ff       	jmp    2020 <_init+0x20>
    206e:	66 90                	xchg   %ax,%ax
    2070:	f3 0f 1e fa          	endbr64
    2074:	68 04 00 00 00       	push   $0x4
    2079:	e9 a2 ff ff ff       	jmp    2020 <_init+0x20>
    207e:	66 90                	xchg   %ax,%ax
    2080:	f3 0f 1e fa          	endbr64
    2084:	68 05 00 00 00       	push   $0x5
    2089:	e9 92 ff ff ff       	jmp    2020 <_init+0x20>
    208e:	66 90                	xchg   %ax,%ax
    2090:	f3 0f 1e fa          	endbr64
    2094:	68 06 00 00 00       	push   $0x6
    2099:	e9 82 ff ff ff       	jmp    2020 <_init+0x20>
    209e:	66 90                	xchg   %ax,%ax
    20a0:	f3 0f 1e fa          	endbr64
    20a4:	68 07 00 00 00       	push   $0x7
    20a9:	e9 72 ff ff ff       	jmp    2020 <_init+0x20>
    20ae:	66 90                	xchg   %ax,%ax
    20b0:	f3 0f 1e fa          	endbr64
    20b4:	68 08 00 00 00       	push   $0x8
    20b9:	e9 62 ff ff ff       	jmp    2020 <_init+0x20>
    20be:	66 90                	xchg   %ax,%ax
    20c0:	f3 0f 1e fa          	endbr64
    20c4:	68 09 00 00 00       	push   $0x9
    20c9:	e9 52 ff ff ff       	jmp    2020 <_init+0x20>
    20ce:	66 90                	xchg   %ax,%ax
    20d0:	f3 0f 1e fa          	endbr64
    20d4:	68 0a 00 00 00       	push   $0xa
    20d9:	e9 42 ff ff ff       	jmp    2020 <_init+0x20>
    20de:	66 90                	xchg   %ax,%ax
    20e0:	f3 0f 1e fa          	endbr64
    20e4:	68 0b 00 00 00       	push   $0xb
    20e9:	e9 32 ff ff ff       	jmp    2020 <_init+0x20>
    20ee:	66 90                	xchg   %ax,%ax
    20f0:	f3 0f 1e fa          	endbr64
    20f4:	68 0c 00 00 00       	push   $0xc
    20f9:	e9 22 ff ff ff       	jmp    2020 <_init+0x20>
    20fe:	66 90                	xchg   %ax,%ax
    2100:	f3 0f 1e fa          	endbr64
    2104:	68 0d 00 00 00       	push   $0xd
    2109:	e9 12 ff ff ff       	jmp    2020 <_init+0x20>
    210e:	66 90                	xchg   %ax,%ax
    2110:	f3 0f 1e fa          	endbr64
    2114:	68 0e 00 00 00       	push   $0xe
    2119:	e9 02 ff ff ff       	jmp    2020 <_init+0x20>
    211e:	66 90                	xchg   %ax,%ax
    2120:	f3 0f 1e fa          	endbr64
    2124:	68 0f 00 00 00       	push   $0xf
    2129:	e9 f2 fe ff ff       	jmp    2020 <_init+0x20>
    212e:	66 90                	xchg   %ax,%ax
    2130:	f3 0f 1e fa          	endbr64
    2134:	68 10 00 00 00       	push   $0x10
    2139:	e9 e2 fe ff ff       	jmp    2020 <_init+0x20>
    213e:	66 90                	xchg   %ax,%ax
    2140:	f3 0f 1e fa          	endbr64
    2144:	68 11 00 00 00       	push   $0x11
    2149:	e9 d2 fe ff ff       	jmp    2020 <_init+0x20>
    214e:	66 90                	xchg   %ax,%ax
    2150:	f3 0f 1e fa          	endbr64
    2154:	68 12 00 00 00       	push   $0x12
    2159:	e9 c2 fe ff ff       	jmp    2020 <_init+0x20>
    215e:	66 90                	xchg   %ax,%ax
    2160:	f3 0f 1e fa          	endbr64
    2164:	68 13 00 00 00       	push   $0x13
    2169:	e9 b2 fe ff ff       	jmp    2020 <_init+0x20>
    216e:	66 90                	xchg   %ax,%ax
    2170:	f3 0f 1e fa          	endbr64
    2174:	68 14 00 00 00       	push   $0x14
    2179:	e9 a2 fe ff ff       	jmp    2020 <_init+0x20>
    217e:	66 90                	xchg   %ax,%ax
    2180:	f3 0f 1e fa          	endbr64
    2184:	68 15 00 00 00       	push   $0x15
    2189:	e9 92 fe ff ff       	jmp    2020 <_init+0x20>
    218e:	66 90                	xchg   %ax,%ax
    2190:	f3 0f 1e fa          	endbr64
    2194:	68 16 00 00 00       	push   $0x16
    2199:	e9 82 fe ff ff       	jmp    2020 <_init+0x20>
    219e:	66 90                	xchg   %ax,%ax
    21a0:	f3 0f 1e fa          	endbr64
    21a4:	68 17 00 00 00       	push   $0x17
    21a9:	e9 72 fe ff ff       	jmp    2020 <_init+0x20>
    21ae:	66 90                	xchg   %ax,%ax
    21b0:	f3 0f 1e fa          	endbr64
    21b4:	68 18 00 00 00       	push   $0x18
    21b9:	e9 62 fe ff ff       	jmp    2020 <_init+0x20>
    21be:	66 90                	xchg   %ax,%ax
    21c0:	f3 0f 1e fa          	endbr64
    21c4:	68 19 00 00 00       	push   $0x19
    21c9:	e9 52 fe ff ff       	jmp    2020 <_init+0x20>
    21ce:	66 90                	xchg   %ax,%ax
    21d0:	f3 0f 1e fa          	endbr64
    21d4:	68 1a 00 00 00       	push   $0x1a
    21d9:	e9 42 fe ff ff       	jmp    2020 <_init+0x20>
    21de:	66 90                	xchg   %ax,%ax
    21e0:	f3 0f 1e fa          	endbr64
    21e4:	68 1b 00 00 00       	push   $0x1b
    21e9:	e9 32 fe ff ff       	jmp    2020 <_init+0x20>
    21ee:	66 90                	xchg   %ax,%ax
    21f0:	f3 0f 1e fa          	endbr64
    21f4:	68 1c 00 00 00       	push   $0x1c
    21f9:	e9 22 fe ff ff       	jmp    2020 <_init+0x20>
    21fe:	66 90                	xchg   %ax,%ax
    2200:	f3 0f 1e fa          	endbr64
    2204:	68 1d 00 00 00       	push   $0x1d
    2209:	e9 12 fe ff ff       	jmp    2020 <_init+0x20>
    220e:	66 90                	xchg   %ax,%ax
    2210:	f3 0f 1e fa          	endbr64
    2214:	68 1e 00 00 00       	push   $0x1e
    2219:	e9 02 fe ff ff       	jmp    2020 <_init+0x20>
    221e:	66 90                	xchg   %ax,%ax
    2220:	f3 0f 1e fa          	endbr64
    2224:	68 1f 00 00 00       	push   $0x1f
    2229:	e9 f2 fd ff ff       	jmp    2020 <_init+0x20>
    222e:	66 90                	xchg   %ax,%ax
    2230:	f3 0f 1e fa          	endbr64
    2234:	68 20 00 00 00       	push   $0x20
    2239:	e9 e2 fd ff ff       	jmp    2020 <_init+0x20>
    223e:	66 90                	xchg   %ax,%ax
    2240:	f3 0f 1e fa          	endbr64
    2244:	68 21 00 00 00       	push   $0x21
    2249:	e9 d2 fd ff ff       	jmp    2020 <_init+0x20>
    224e:	66 90                	xchg   %ax,%ax
    2250:	f3 0f 1e fa          	endbr64
    2254:	68 22 00 00 00       	push   $0x22
    2259:	e9 c2 fd ff ff       	jmp    2020 <_init+0x20>
    225e:	66 90                	xchg   %ax,%ax
    2260:	f3 0f 1e fa          	endbr64
    2264:	68 23 00 00 00       	push   $0x23
    2269:	e9 b2 fd ff ff       	jmp    2020 <_init+0x20>
    226e:	66 90                	xchg   %ax,%ax
    2270:	f3 0f 1e fa          	endbr64
    2274:	68 24 00 00 00       	push   $0x24
    2279:	e9 a2 fd ff ff       	jmp    2020 <_init+0x20>
    227e:	66 90                	xchg   %ax,%ax
    2280:	f3 0f 1e fa          	endbr64
    2284:	68 25 00 00 00       	push   $0x25
    2289:	e9 92 fd ff ff       	jmp    2020 <_init+0x20>
    228e:	66 90                	xchg   %ax,%ax
    2290:	f3 0f 1e fa          	endbr64
    2294:	68 26 00 00 00       	push   $0x26
    2299:	e9 82 fd ff ff       	jmp    2020 <_init+0x20>
    229e:	66 90                	xchg   %ax,%ax
    22a0:	f3 0f 1e fa          	endbr64
    22a4:	68 27 00 00 00       	push   $0x27
    22a9:	e9 72 fd ff ff       	jmp    2020 <_init+0x20>
    22ae:	66 90                	xchg   %ax,%ax
    22b0:	f3 0f 1e fa          	endbr64
    22b4:	68 28 00 00 00       	push   $0x28
    22b9:	e9 62 fd ff ff       	jmp    2020 <_init+0x20>
    22be:	66 90                	xchg   %ax,%ax
    22c0:	f3 0f 1e fa          	endbr64
    22c4:	68 29 00 00 00       	push   $0x29
    22c9:	e9 52 fd ff ff       	jmp    2020 <_init+0x20>
    22ce:	66 90                	xchg   %ax,%ax
    22d0:	f3 0f 1e fa          	endbr64
    22d4:	68 2a 00 00 00       	push   $0x2a
    22d9:	e9 42 fd ff ff       	jmp    2020 <_init+0x20>
    22de:	66 90                	xchg   %ax,%ax
    22e0:	f3 0f 1e fa          	endbr64
    22e4:	68 2b 00 00 00       	push   $0x2b
    22e9:	e9 32 fd ff ff       	jmp    2020 <_init+0x20>
    22ee:	66 90                	xchg   %ax,%ax
    22f0:	f3 0f 1e fa          	endbr64
    22f4:	68 2c 00 00 00       	push   $0x2c
    22f9:	e9 22 fd ff ff       	jmp    2020 <_init+0x20>
    22fe:	66 90                	xchg   %ax,%ax
    2300:	f3 0f 1e fa          	endbr64
    2304:	68 2d 00 00 00       	push   $0x2d
    2309:	e9 12 fd ff ff       	jmp    2020 <_init+0x20>
    230e:	66 90                	xchg   %ax,%ax
    2310:	f3 0f 1e fa          	endbr64
    2314:	68 2e 00 00 00       	push   $0x2e
    2319:	e9 02 fd ff ff       	jmp    2020 <_init+0x20>
    231e:	66 90                	xchg   %ax,%ax
    2320:	f3 0f 1e fa          	endbr64
    2324:	68 2f 00 00 00       	push   $0x2f
    2329:	e9 f2 fc ff ff       	jmp    2020 <_init+0x20>
    232e:	66 90                	xchg   %ax,%ax
    2330:	f3 0f 1e fa          	endbr64
    2334:	68 30 00 00 00       	push   $0x30
    2339:	e9 e2 fc ff ff       	jmp    2020 <_init+0x20>
    233e:	66 90                	xchg   %ax,%ax
    2340:	f3 0f 1e fa          	endbr64
    2344:	68 31 00 00 00       	push   $0x31
    2349:	e9 d2 fc ff ff       	jmp    2020 <_init+0x20>
    234e:	66 90                	xchg   %ax,%ax

Disassembly of section .plt.got:

0000000000002350 <__cxa_finalize@plt>:
    2350:	f3 0f 1e fa          	endbr64
    2354:	ff 25 6e 6c 00 00    	jmp    *0x6c6e(%rip)        # 8fc8 <__cxa_finalize@GLIBC_2.2.5>
    235a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

0000000000002360 <_Znam@plt>:
    2360:	f3 0f 1e fa          	endbr64
    2364:	ff 25 ce 6a 00 00    	jmp    *0x6ace(%rip)        # 8e38 <_Znam@GLIBCXX_3.4>
    236a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002370 <curandDestroyGenerator@plt>:
    2370:	f3 0f 1e fa          	endbr64
    2374:	ff 25 c6 6a 00 00    	jmp    *0x6ac6(%rip)        # 8e40 <curandDestroyGenerator@libcurand.so.10>
    237a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002380 <cudaMalloc@plt>:
    2380:	f3 0f 1e fa          	endbr64
    2384:	ff 25 be 6a 00 00    	jmp    *0x6abe(%rip)        # 8e48 <cudaMalloc@libcudart.so.12>
    238a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002390 <cudaMemset@plt>:
    2390:	f3 0f 1e fa          	endbr64
    2394:	ff 25 b6 6a 00 00    	jmp    *0x6ab6(%rip)        # 8e50 <cudaMemset@libcudart.so.12>
    239a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000023a0 <_ZNSaIcED2Ev@plt>:
    23a0:	f3 0f 1e fa          	endbr64
    23a4:	ff 25 ae 6a 00 00    	jmp    *0x6aae(%rip)        # 8e58 <_ZNSaIcED2Ev@GLIBCXX_3.4>
    23aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000023b0 <cudaEventCreate@plt>:
    23b0:	f3 0f 1e fa          	endbr64
    23b4:	ff 25 a6 6a 00 00    	jmp    *0x6aa6(%rip)        # 8e60 <cudaEventCreate@libcudart.so.12>
    23ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000023c0 <curandSetPseudoRandomGeneratorSeed@plt>:
    23c0:	f3 0f 1e fa          	endbr64
    23c4:	ff 25 9e 6a 00 00    	jmp    *0x6a9e(%rip)        # 8e68 <curandSetPseudoRandomGeneratorSeed@libcurand.so.10>
    23ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000023d0 <__cudaUnregisterFatBinary@plt>:
    23d0:	f3 0f 1e fa          	endbr64
    23d4:	ff 25 96 6a 00 00    	jmp    *0x6a96(%rip)        # 8e70 <__cudaUnregisterFatBinary@libcudart.so.12>
    23da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000023e0 <_ZSt17__throw_bad_allocv@plt>:
    23e0:	f3 0f 1e fa          	endbr64
    23e4:	ff 25 8e 6a 00 00    	jmp    *0x6a8e(%rip)        # 8e78 <_ZSt17__throw_bad_allocv@GLIBCXX_3.4>
    23ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000023f0 <__cxa_begin_catch@plt>:
    23f0:	f3 0f 1e fa          	endbr64
    23f4:	ff 25 86 6a 00 00    	jmp    *0x6a86(%rip)        # 8e80 <__cxa_begin_catch@CXXABI_1.3>
    23fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002400 <_ZSt20__throw_length_errorPKc@plt>:
    2400:	f3 0f 1e fa          	endbr64
    2404:	ff 25 7e 6a 00 00    	jmp    *0x6a7e(%rip)        # 8e88 <_ZSt20__throw_length_errorPKc@GLIBCXX_3.4>
    240a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002410 <cublasGetMathMode@plt>:
    2410:	f3 0f 1e fa          	endbr64
    2414:	ff 25 76 6a 00 00    	jmp    *0x6a76(%rip)        # 8e90 <cublasGetMathMode@libcublas.so.12>
    241a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002420 <cudaEventDestroy@plt>:
    2420:	f3 0f 1e fa          	endbr64
    2424:	ff 25 6e 6a 00 00    	jmp    *0x6a6e(%rip)        # 8e98 <cudaEventDestroy@libcudart.so.12>
    242a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002430 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@plt>:
    2430:	f3 0f 1e fa          	endbr64
    2434:	ff 25 66 6a 00 00    	jmp    *0x6a66(%rip)        # 8ea0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@GLIBCXX_3.4.21>
    243a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002440 <cudaEventRecord@plt>:
    2440:	f3 0f 1e fa          	endbr64
    2444:	ff 25 5e 6a 00 00    	jmp    *0x6a5e(%rip)        # 8ea8 <cudaEventRecord@libcudart.so.12>
    244a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002450 <_ZNSolsEf@plt>:
    2450:	f3 0f 1e fa          	endbr64
    2454:	ff 25 56 6a 00 00    	jmp    *0x6a56(%rip)        # 8eb0 <_ZNSolsEf@GLIBCXX_3.4>
    245a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002460 <cublasGemmEx@plt>:
    2460:	f3 0f 1e fa          	endbr64
    2464:	ff 25 4e 6a 00 00    	jmp    *0x6a4e(%rip)        # 8eb8 <cublasGemmEx@libcublas.so.12>
    246a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002470 <_ZSt28__throw_bad_array_new_lengthv@plt>:
    2470:	f3 0f 1e fa          	endbr64
    2474:	ff 25 46 6a 00 00    	jmp    *0x6a46(%rip)        # 8ec0 <_ZSt28__throw_bad_array_new_lengthv@GLIBCXX_3.4.29>
    247a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002480 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv@plt>:
    2480:	f3 0f 1e fa          	endbr64
    2484:	ff 25 3e 6a 00 00    	jmp    *0x6a3e(%rip)        # 8ec8 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv@GLIBCXX_3.4.21>
    248a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002490 <__cxa_atexit@plt>:
    2490:	f3 0f 1e fa          	endbr64
    2494:	ff 25 36 6a 00 00    	jmp    *0x6a36(%rip)        # 8ed0 <__cxa_atexit@GLIBC_2.2.5>
    249a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000024a0 <_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@plt>:
    24a0:	f3 0f 1e fa          	endbr64
    24a4:	ff 25 2e 6a 00 00    	jmp    *0x6a2e(%rip)        # 8ed8 <_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@GLIBCXX_3.4.21>
    24aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000024b0 <__cudaRegisterFatBinary@plt>:
    24b0:	f3 0f 1e fa          	endbr64
    24b4:	ff 25 26 6a 00 00    	jmp    *0x6a26(%rip)        # 8ee0 <__cudaRegisterFatBinary@libcudart.so.12>
    24ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000024c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>:
    24c0:	f3 0f 1e fa          	endbr64
    24c4:	ff 25 1e 6a 00 00    	jmp    *0x6a1e(%rip)        # 8ee8 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@GLIBCXX_3.4>
    24ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000024d0 <_Znwm@plt>:
    24d0:	f3 0f 1e fa          	endbr64
    24d4:	ff 25 16 6a 00 00    	jmp    *0x6a16(%rip)        # 8ef0 <_Znwm@GLIBCXX_3.4>
    24da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000024e0 <cudaEventElapsedTime@plt>:
    24e0:	f3 0f 1e fa          	endbr64
    24e4:	ff 25 0e 6a 00 00    	jmp    *0x6a0e(%rip)        # 8ef8 <cudaEventElapsedTime@libcudart.so.12>
    24ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000024f0 <_ZdlPvm@plt>:
    24f0:	f3 0f 1e fa          	endbr64
    24f4:	ff 25 06 6a 00 00    	jmp    *0x6a06(%rip)        # 8f00 <_ZdlPvm@CXXABI_1.3.9>
    24fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002500 <_ZNSolsEPFRSoS_E@plt>:
    2500:	f3 0f 1e fa          	endbr64
    2504:	ff 25 fe 69 00 00    	jmp    *0x69fe(%rip)        # 8f08 <_ZNSolsEPFRSoS_E@GLIBCXX_3.4>
    250a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002510 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcRKS3_@plt>:
    2510:	f3 0f 1e fa          	endbr64
    2514:	ff 25 f6 69 00 00    	jmp    *0x69f6(%rip)        # 8f10 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcRKS3_@GLIBCXX_3.4.21>
    251a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002520 <_ZNSaIcED1Ev@plt>:
    2520:	f3 0f 1e fa          	endbr64
    2524:	ff 25 ee 69 00 00    	jmp    *0x69ee(%rip)        # 8f18 <_ZNSaIcED1Ev@GLIBCXX_3.4>
    252a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002530 <__stack_chk_fail@plt>:
    2530:	f3 0f 1e fa          	endbr64
    2534:	ff 25 e6 69 00 00    	jmp    *0x69e6(%rip)        # 8f20 <__stack_chk_fail@GLIBC_2.4>
    253a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002540 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@plt>:
    2540:	f3 0f 1e fa          	endbr64
    2544:	ff 25 de 69 00 00    	jmp    *0x69de(%rip)        # 8f28 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@GLIBCXX_3.4.21>
    254a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002550 <cudaMemcpy@plt>:
    2550:	f3 0f 1e fa          	endbr64
    2554:	ff 25 d6 69 00 00    	jmp    *0x69d6(%rip)        # 8f30 <cudaMemcpy@libcudart.so.12>
    255a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002560 <cublasDestroy_v2@plt>:
    2560:	f3 0f 1e fa          	endbr64
    2564:	ff 25 ce 69 00 00    	jmp    *0x69ce(%rip)        # 8f38 <cublasDestroy_v2@libcublas.so.12>
    256a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002570 <curandCreateGenerator@plt>:
    2570:	f3 0f 1e fa          	endbr64
    2574:	ff 25 c6 69 00 00    	jmp    *0x69c6(%rip)        # 8f40 <curandCreateGenerator@libcurand.so.10>
    257a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002580 <_ZdaPv@plt>:
    2580:	f3 0f 1e fa          	endbr64
    2584:	ff 25 be 69 00 00    	jmp    *0x69be(%rip)        # 8f48 <_ZdaPv@GLIBCXX_3.4>
    258a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002590 <__cxa_throw_bad_array_new_length@plt>:
    2590:	f3 0f 1e fa          	endbr64
    2594:	ff 25 b6 69 00 00    	jmp    *0x69b6(%rip)        # 8f50 <__cxa_throw_bad_array_new_length@CXXABI_1.3.8>
    259a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000025a0 <cudaFree@plt>:
    25a0:	f3 0f 1e fa          	endbr64
    25a4:	ff 25 ae 69 00 00    	jmp    *0x69ae(%rip)        # 8f58 <cudaFree@libcudart.so.12>
    25aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000025b0 <_ZNSt8ios_base4InitC1Ev@plt>:
    25b0:	f3 0f 1e fa          	endbr64
    25b4:	ff 25 a6 69 00 00    	jmp    *0x69a6(%rip)        # 8f60 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
    25ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000025c0 <cublasCreate_v2@plt>:
    25c0:	f3 0f 1e fa          	endbr64
    25c4:	ff 25 9e 69 00 00    	jmp    *0x699e(%rip)        # 8f68 <cublasCreate_v2@libcublas.so.12>
    25ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000025d0 <cudaEventSynchronize@plt>:
    25d0:	f3 0f 1e fa          	endbr64
    25d4:	ff 25 96 69 00 00    	jmp    *0x6996(%rip)        # 8f70 <cudaEventSynchronize@libcudart.so.12>
    25da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000025e0 <memmove@plt>:
    25e0:	f3 0f 1e fa          	endbr64
    25e4:	ff 25 8e 69 00 00    	jmp    *0x698e(%rip)        # 8f78 <memmove@GLIBC_2.2.5>
    25ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000025f0 <__cxa_end_catch@plt>:
    25f0:	f3 0f 1e fa          	endbr64
    25f4:	ff 25 86 69 00 00    	jmp    *0x6986(%rip)        # 8f80 <__cxa_end_catch@CXXABI_1.3>
    25fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002600 <cudaDeviceSynchronize@plt>:
    2600:	f3 0f 1e fa          	endbr64
    2604:	ff 25 7e 69 00 00    	jmp    *0x697e(%rip)        # 8f88 <cudaDeviceSynchronize@libcudart.so.12>
    260a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002610 <__cudaInitModule@plt>:
    2610:	f3 0f 1e fa          	endbr64
    2614:	ff 25 76 69 00 00    	jmp    *0x6976(%rip)        # 8f90 <__cudaInitModule@libcudart.so.12>
    261a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002620 <_ZNSolsEi@plt>:
    2620:	f3 0f 1e fa          	endbr64
    2624:	ff 25 6e 69 00 00    	jmp    *0x696e(%rip)        # 8f98 <_ZNSolsEi@GLIBCXX_3.4>
    262a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002630 <_Unwind_Resume@plt>:
    2630:	f3 0f 1e fa          	endbr64
    2634:	ff 25 66 69 00 00    	jmp    *0x6966(%rip)        # 8fa0 <_Unwind_Resume@GCC_3.0>
    263a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002640 <_ZNSaIcEC1Ev@plt>:
    2640:	f3 0f 1e fa          	endbr64
    2644:	ff 25 5e 69 00 00    	jmp    *0x695e(%rip)        # 8fa8 <_ZNSaIcEC1Ev@GLIBCXX_3.4>
    264a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002650 <__cudaRegisterFatBinaryEnd@plt>:
    2650:	f3 0f 1e fa          	endbr64
    2654:	ff 25 56 69 00 00    	jmp    *0x6956(%rip)        # 8fb0 <__cudaRegisterFatBinaryEnd@libcudart.so.12>
    265a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002660 <curandGenerateNormal@plt>:
    2660:	f3 0f 1e fa          	endbr64
    2664:	ff 25 4e 69 00 00    	jmp    *0x694e(%rip)        # 8fb8 <curandGenerateNormal@libcurand.so.10>
    266a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002670 <_ZNSolsEd@plt>:
    2670:	f3 0f 1e fa          	endbr64
    2674:	ff 25 46 69 00 00    	jmp    *0x6946(%rip)        # 8fc0 <_ZNSolsEd@GLIBCXX_3.4>
    267a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

Disassembly of section .text:

0000000000002680 <_start>:
    2680:	f3 0f 1e fa          	endbr64
    2684:	31 ed                	xor    %ebp,%ebp
    2686:	49 89 d1             	mov    %rdx,%r9
    2689:	5e                   	pop    %rsi
    268a:	48 89 e2             	mov    %rsp,%rdx
    268d:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    2691:	50                   	push   %rax
    2692:	54                   	push   %rsp
    2693:	45 31 c0             	xor    %r8d,%r8d
    2696:	31 c9                	xor    %ecx,%ecx
    2698:	48 8d 3d 80 0a 00 00 	lea    0xa80(%rip),%rdi        # 311f <main>
    269f:	ff 15 33 69 00 00    	call   *0x6933(%rip)        # 8fd8 <__libc_start_main@GLIBC_2.34>
    26a5:	f4                   	hlt
    26a6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    26ad:	00 00 00 

00000000000026b0 <deregister_tm_clones>:
    26b0:	48 8d 3d 61 69 00 00 	lea    0x6961(%rip),%rdi        # 9018 <__TMC_END__>
    26b7:	48 8d 05 5a 69 00 00 	lea    0x695a(%rip),%rax        # 9018 <__TMC_END__>
    26be:	48 39 f8             	cmp    %rdi,%rax
    26c1:	74 15                	je     26d8 <deregister_tm_clones+0x28>
    26c3:	48 8b 05 16 69 00 00 	mov    0x6916(%rip),%rax        # 8fe0 <_ITM_deregisterTMCloneTable@Base>
    26ca:	48 85 c0             	test   %rax,%rax
    26cd:	74 09                	je     26d8 <deregister_tm_clones+0x28>
    26cf:	ff e0                	jmp    *%rax
    26d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    26d8:	c3                   	ret
    26d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000026e0 <register_tm_clones>:
    26e0:	48 8d 3d 31 69 00 00 	lea    0x6931(%rip),%rdi        # 9018 <__TMC_END__>
    26e7:	48 8d 35 2a 69 00 00 	lea    0x692a(%rip),%rsi        # 9018 <__TMC_END__>
    26ee:	48 29 fe             	sub    %rdi,%rsi
    26f1:	48 89 f0             	mov    %rsi,%rax
    26f4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    26f8:	48 c1 f8 03          	sar    $0x3,%rax
    26fc:	48 01 c6             	add    %rax,%rsi
    26ff:	48 d1 fe             	sar    $1,%rsi
    2702:	74 14                	je     2718 <register_tm_clones+0x38>
    2704:	48 8b 05 e5 68 00 00 	mov    0x68e5(%rip),%rax        # 8ff0 <_ITM_registerTMCloneTable@Base>
    270b:	48 85 c0             	test   %rax,%rax
    270e:	74 08                	je     2718 <register_tm_clones+0x38>
    2710:	ff e0                	jmp    *%rax
    2712:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2718:	c3                   	ret
    2719:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002720 <__do_global_dtors_aux>:
    2720:	f3 0f 1e fa          	endbr64
    2724:	80 3d 25 6a 00 00 00 	cmpb   $0x0,0x6a25(%rip)        # 9150 <completed.0>
    272b:	75 2b                	jne    2758 <__do_global_dtors_aux+0x38>
    272d:	55                   	push   %rbp
    272e:	48 83 3d 92 68 00 00 	cmpq   $0x0,0x6892(%rip)        # 8fc8 <__cxa_finalize@GLIBC_2.2.5>
    2735:	00 
    2736:	48 89 e5             	mov    %rsp,%rbp
    2739:	74 0c                	je     2747 <__do_global_dtors_aux+0x27>
    273b:	48 8b 3d c6 68 00 00 	mov    0x68c6(%rip),%rdi        # 9008 <__dso_handle>
    2742:	e8 09 fc ff ff       	call   2350 <__cxa_finalize@plt>
    2747:	e8 64 ff ff ff       	call   26b0 <deregister_tm_clones>
    274c:	c6 05 fd 69 00 00 01 	movb   $0x1,0x69fd(%rip)        # 9150 <completed.0>
    2753:	5d                   	pop    %rbp
    2754:	c3                   	ret
    2755:	0f 1f 00             	nopl   (%rax)
    2758:	c3                   	ret
    2759:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002760 <frame_dummy>:
    2760:	f3 0f 1e fa          	endbr64
    2764:	e9 77 ff ff ff       	jmp    26e0 <register_tm_clones>

0000000000002769 <_ZL37__nv_save_fatbinhandle_for_managed_rtPPv>:
    2769:	f3 0f 1e fa          	endbr64
    276d:	55                   	push   %rbp
    276e:	48 89 e5             	mov    %rsp,%rbp
    2771:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    2775:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    2779:	48 89 05 e0 69 00 00 	mov    %rax,0x69e0(%rip)        # 9160 <_ZL32__nv_fatbinhandle_for_managed_rt>
    2780:	90                   	nop
    2781:	5d                   	pop    %rbp
    2782:	c3                   	ret

0000000000002783 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t>:
    2783:	55                   	push   %rbp
    2784:	48 89 e5             	mov    %rsp,%rbp
    2787:	48 83 ec 40          	sub    $0x40,%rsp
    278b:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    278f:	89 75 d4             	mov    %esi,-0x2c(%rbp)
    2792:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
    2796:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    279d:	00 00 
    279f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    27a3:	31 c0                	xor    %eax,%eax
    27a5:	c7 45 f0 00 00 00 00 	movl   $0x0,-0x10(%rbp)
    27ac:	c7 45 f4 00 00 00 00 	movl   $0x0,-0xc(%rbp)
    27b3:	48 8d 55 f0          	lea    -0x10(%rbp),%rdx
    27b7:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    27bb:	48 89 d6             	mov    %rdx,%rsi
    27be:	48 89 c7             	mov    %rax,%rdi
    27c1:	e8 4a fc ff ff       	call   2410 <cublasGetMathMode@plt>
    27c6:	89 45 f4             	mov    %eax,-0xc(%rbp)
    27c9:	83 7d f4 00          	cmpl   $0x0,-0xc(%rbp)
    27cd:	74 08                	je     27d7 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x54>
    27cf:	8b 45 f4             	mov    -0xc(%rbp),%eax
    27d2:	e9 bf 00 00 00       	jmp    2896 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x113>
    27d7:	8b 45 f0             	mov    -0x10(%rbp),%eax
    27da:	83 e0 0f             	and    $0xf,%eax
    27dd:	83 f8 02             	cmp    $0x2,%eax
    27e0:	0f 94 c0             	sete   %al
    27e3:	88 45 ef             	mov    %al,-0x11(%rbp)
    27e6:	8b 45 d4             	mov    -0x2c(%rbp),%eax
    27e9:	83 f8 0a             	cmp    $0xa,%eax
    27ec:	0f 87 9f 00 00 00    	ja     2891 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x10e>
    27f2:	89 c0                	mov    %eax,%eax
    27f4:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    27fb:	00 
    27fc:	48 8d 05 35 39 00 00 	lea    0x3935(%rip),%rax        # 6138 <_ZN2nv6targetL5sm_90E+0x8>
    2803:	8b 04 02             	mov    (%rdx,%rax,1),%eax
    2806:	48 98                	cltq
    2808:	48 8d 15 29 39 00 00 	lea    0x3929(%rip),%rdx        # 6138 <_ZN2nv6targetL5sm_90E+0x8>
    280f:	48 01 d0             	add    %rdx,%rax
    2812:	3e ff e0             	notrack jmp *%rax
    2815:	80 7d ef 00          	cmpb   $0x0,-0x11(%rbp)
    2819:	74 07                	je     2822 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x9f>
    281b:	ba 45 00 00 00       	mov    $0x45,%edx
    2820:	eb 05                	jmp    2827 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0xa4>
    2822:	ba 44 00 00 00       	mov    $0x44,%edx
    2827:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    282b:	89 10                	mov    %edx,(%rax)
    282d:	b8 00 00 00 00       	mov    $0x0,%eax
    2832:	eb 62                	jmp    2896 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x113>
    2834:	80 7d ef 00          	cmpb   $0x0,-0x11(%rbp)
    2838:	74 07                	je     2841 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0xbe>
    283a:	ba 47 00 00 00       	mov    $0x47,%edx
    283f:	eb 05                	jmp    2846 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0xc3>
    2841:	ba 46 00 00 00       	mov    $0x46,%edx
    2846:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    284a:	89 10                	mov    %edx,(%rax)
    284c:	b8 00 00 00 00       	mov    $0x0,%eax
    2851:	eb 43                	jmp    2896 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x113>
    2853:	80 7d ef 00          	cmpb   $0x0,-0x11(%rbp)
    2857:	74 07                	je     2860 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0xdd>
    2859:	ba 41 00 00 00       	mov    $0x41,%edx
    285e:	eb 05                	jmp    2865 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0xe2>
    2860:	ba 40 00 00 00       	mov    $0x40,%edx
    2865:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    2869:	89 10                	mov    %edx,(%rax)
    286b:	b8 00 00 00 00       	mov    $0x0,%eax
    2870:	eb 24                	jmp    2896 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x113>
    2872:	80 7d ef 00          	cmpb   $0x0,-0x11(%rbp)
    2876:	74 07                	je     287f <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0xfc>
    2878:	ba 49 00 00 00       	mov    $0x49,%edx
    287d:	eb 05                	jmp    2884 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x101>
    287f:	ba 48 00 00 00       	mov    $0x48,%edx
    2884:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    2888:	89 10                	mov    %edx,(%rax)
    288a:	b8 00 00 00 00       	mov    $0x0,%eax
    288f:	eb 05                	jmp    2896 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x113>
    2891:	b8 0f 00 00 00       	mov    $0xf,%eax
    2896:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    289a:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    28a1:	00 00 
    28a3:	74 05                	je     28aa <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t+0x127>
    28a5:	e8 86 fc ff ff       	call   2530 <__stack_chk_fail@plt>
    28aa:	c9                   	leave
    28ab:	c3                   	ret

00000000000028ac <_ZL12cublasGemmExP13cublasContext17cublasOperation_tS1_iiiPKvS3_14cudaDataType_tiS3_S4_iS3_PvS4_iS4_16cublasGemmAlgo_t>:
    28ac:	55                   	push   %rbp
    28ad:	48 89 e5             	mov    %rsp,%rbp
    28b0:	48 83 ec 60          	sub    $0x60,%rsp
    28b4:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    28b8:	89 75 e4             	mov    %esi,-0x1c(%rbp)
    28bb:	89 55 e0             	mov    %edx,-0x20(%rbp)
    28be:	89 4d dc             	mov    %ecx,-0x24(%rbp)
    28c1:	44 89 45 d8          	mov    %r8d,-0x28(%rbp)
    28c5:	44 89 4d d4          	mov    %r9d,-0x2c(%rbp)
    28c9:	48 8b 45 10          	mov    0x10(%rbp),%rax
    28cd:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    28d1:	48 8b 45 18          	mov    0x18(%rbp),%rax
    28d5:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
    28d9:	48 8b 45 30          	mov    0x30(%rbp),%rax
    28dd:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
    28e1:	48 8b 45 48          	mov    0x48(%rbp),%rax
    28e5:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
    28e9:	48 8b 45 50          	mov    0x50(%rbp),%rax
    28ed:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
    28f1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    28f8:	00 00 
    28fa:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    28fe:	31 c0                	xor    %eax,%eax
    2900:	c7 45 f0 44 00 00 00 	movl   $0x44,-0x10(%rbp)
    2907:	c7 45 f4 00 00 00 00 	movl   $0x0,-0xc(%rbp)
    290e:	48 8d 55 f0          	lea    -0x10(%rbp),%rdx
    2912:	8b 4d 68             	mov    0x68(%rbp),%ecx
    2915:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    2919:	89 ce                	mov    %ecx,%esi
    291b:	48 89 c7             	mov    %rax,%rdi
    291e:	e8 60 fe ff ff       	call   2783 <_ZL24cublasMigrateComputeTypeP13cublasContext14cudaDataType_tP19cublasComputeType_t>
    2923:	89 45 f4             	mov    %eax,-0xc(%rbp)
    2926:	83 7d f4 00          	cmpl   $0x0,-0xc(%rbp)
    292a:	74 05                	je     2931 <_ZL12cublasGemmExP13cublasContext17cublasOperation_tS1_iiiPKvS3_14cudaDataType_tiS3_S4_iS3_PvS4_iS4_16cublasGemmAlgo_t+0x85>
    292c:	8b 45 f4             	mov    -0xc(%rbp),%eax
    292f:	eb 5a                	jmp    298b <_ZL12cublasGemmExP13cublasContext17cublasOperation_tS1_iiiPKvS3_14cudaDataType_tiS3_S4_iS3_PvS4_iS4_16cublasGemmAlgo_t+0xdf>
    2931:	8b 7d f0             	mov    -0x10(%rbp),%edi
    2934:	44 8b 4d d4          	mov    -0x2c(%rbp),%r9d
    2938:	44 8b 55 d8          	mov    -0x28(%rbp),%r10d
    293c:	8b 4d dc             	mov    -0x24(%rbp),%ecx
    293f:	8b 55 e0             	mov    -0x20(%rbp),%edx
    2942:	8b 75 e4             	mov    -0x1c(%rbp),%esi
    2945:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    2949:	48 83 ec 08          	sub    $0x8,%rsp
    294d:	44 8b 45 70          	mov    0x70(%rbp),%r8d
    2951:	41 50                	push   %r8
    2953:	57                   	push   %rdi
    2954:	8b 7d 60             	mov    0x60(%rbp),%edi
    2957:	57                   	push   %rdi
    2958:	8b 7d 58             	mov    0x58(%rbp),%edi
    295b:	57                   	push   %rdi
    295c:	ff 75 a8             	push   -0x58(%rbp)
    295f:	ff 75 b0             	push   -0x50(%rbp)
    2962:	8b 7d 40             	mov    0x40(%rbp),%edi
    2965:	57                   	push   %rdi
    2966:	8b 7d 38             	mov    0x38(%rbp),%edi
    2969:	57                   	push   %rdi
    296a:	ff 75 b8             	push   -0x48(%rbp)
    296d:	8b 7d 28             	mov    0x28(%rbp),%edi
    2970:	57                   	push   %rdi
    2971:	8b 7d 20             	mov    0x20(%rbp),%edi
    2974:	57                   	push   %rdi
    2975:	ff 75 c0             	push   -0x40(%rbp)
    2978:	ff 75 c8             	push   -0x38(%rbp)
    297b:	45 89 d0             	mov    %r10d,%r8d
    297e:	48 89 c7             	mov    %rax,%rdi
    2981:	e8 da fa ff ff       	call   2460 <cublasGemmEx@plt>
    2986:	48 83 c4 70          	add    $0x70,%rsp
    298a:	90                   	nop
    298b:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    298f:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    2996:	00 00 
    2998:	74 05                	je     299f <_ZL12cublasGemmExP13cublasContext17cublasOperation_tS1_iiiPKvS3_14cudaDataType_tiS3_S4_iS3_PvS4_iS4_16cublasGemmAlgo_t+0xf3>
    299a:	e8 91 fb ff ff       	call   2530 <__stack_chk_fail@plt>
    299f:	c9                   	leave
    29a0:	c3                   	ret

00000000000029a1 <_Z10elapsed_msP10CUevent_stS0_>:
    29a1:	f3 0f 1e fa          	endbr64
    29a5:	55                   	push   %rbp
    29a6:	48 89 e5             	mov    %rsp,%rbp
    29a9:	48 83 ec 20          	sub    $0x20,%rsp
    29ad:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    29b1:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    29b5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    29bc:	00 00 
    29be:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    29c2:	31 c0                	xor    %eax,%eax
    29c4:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    29c8:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
    29cc:	48 8d 45 f4          	lea    -0xc(%rbp),%rax
    29d0:	48 89 ce             	mov    %rcx,%rsi
    29d3:	48 89 c7             	mov    %rax,%rdi
    29d6:	e8 05 fb ff ff       	call   24e0 <cudaEventElapsedTime@plt>
    29db:	f3 0f 10 45 f4       	movss  -0xc(%rbp),%xmm0
    29e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    29e4:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    29eb:	00 00 
    29ed:	74 05                	je     29f4 <_Z10elapsed_msP10CUevent_stS0_+0x53>
    29ef:	e8 3c fb ff ff       	call   2530 <__stack_chk_fail@plt>
    29f4:	c9                   	leave
    29f5:	c3                   	ret

00000000000029f6 <_Z17initialize_matrixP6__halfiiy>:
    29f6:	f3 0f 1e fa          	endbr64
    29fa:	55                   	push   %rbp
    29fb:	48 89 e5             	mov    %rsp,%rbp
    29fe:	48 83 ec 40          	sub    $0x40,%rsp
    2a02:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    2a06:	89 75 d4             	mov    %esi,-0x2c(%rbp)
    2a09:	89 55 d0             	mov    %edx,-0x30(%rbp)
    2a0c:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
    2a10:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2a17:	00 00 
    2a19:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    2a1d:	31 c0                	xor    %eax,%eax
    2a1f:	48 8d 45 e8          	lea    -0x18(%rbp),%rax
    2a23:	be 64 00 00 00       	mov    $0x64,%esi
    2a28:	48 89 c7             	mov    %rax,%rdi
    2a2b:	e8 40 fb ff ff       	call   2570 <curandCreateGenerator@plt>
    2a30:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    2a34:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
    2a38:	48 89 d6             	mov    %rdx,%rsi
    2a3b:	48 89 c7             	mov    %rax,%rdi
    2a3e:	e8 7d f9 ff ff       	call   23c0 <curandSetPseudoRandomGeneratorSeed@plt>
    2a43:	8b 45 d4             	mov    -0x2c(%rbp),%eax
    2a46:	0f af 45 d0          	imul   -0x30(%rbp),%eax
    2a4a:	48 98                	cltq
    2a4c:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    2a53:	00 
    2a54:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    2a58:	48 89 d6             	mov    %rdx,%rsi
    2a5b:	48 89 c7             	mov    %rax,%rdi
    2a5e:	e8 51 12 00 00       	call   3cb4 <_Z10cudaMallocIfE9cudaErrorPPT_m>
    2a63:	8b 45 d4             	mov    -0x2c(%rbp),%eax
    2a66:	0f af 45 d0          	imul   -0x30(%rbp),%eax
    2a6a:	48 63 d0             	movslq %eax,%rdx
    2a6d:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    2a71:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    2a75:	f3 0f 10 0d 9f 39 00 	movss  0x399f(%rip),%xmm1        # 641c <_ZSt12__is_ratio_vISt5ratioILl1000000000ELl1EEE+0x3>
    2a7c:	00 
    2a7d:	8b 35 9d 39 00 00    	mov    0x399d(%rip),%esi        # 6420 <_ZSt12__is_ratio_vISt5ratioILl1000000000ELl1EEE+0x7>
    2a83:	66 0f 6e c6          	movd   %esi,%xmm0
    2a87:	48 89 ce             	mov    %rcx,%rsi
    2a8a:	48 89 c7             	mov    %rax,%rdi
    2a8d:	e8 ce fb ff ff       	call   2660 <curandGenerateNormal@plt>
    2a92:	8b 45 d4             	mov    -0x2c(%rbp),%eax
    2a95:	0f af 45 d0          	imul   -0x30(%rbp),%eax
    2a99:	48 98                	cltq
    2a9b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    2aa2:	00 
    2aa3:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
    2aa7:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    2aab:	b9 03 00 00 00       	mov    $0x3,%ecx
    2ab0:	48 89 c7             	mov    %rax,%rdi
    2ab3:	e8 98 fa ff ff       	call   2550 <cudaMemcpy@plt>
    2ab8:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    2abc:	48 89 c7             	mov    %rax,%rdi
    2abf:	e8 dc fa ff ff       	call   25a0 <cudaFree@plt>
    2ac4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    2ac8:	48 89 c7             	mov    %rax,%rdi
    2acb:	e8 a0 f8 ff ff       	call   2370 <curandDestroyGenerator@plt>
    2ad0:	90                   	nop
    2ad1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    2ad5:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    2adc:	00 00 
    2ade:	74 05                	je     2ae5 <_Z17initialize_matrixP6__halfiiy+0xef>
    2ae0:	e8 4b fa ff ff       	call   2530 <__stack_chk_fail@plt>
    2ae5:	c9                   	leave
    2ae6:	c3                   	ret

0000000000002ae7 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult>:
    2ae7:	f3 0f 1e fa          	endbr64
    2aeb:	55                   	push   %rbp
    2aec:	48 89 e5             	mov    %rsp,%rbp
    2aef:	53                   	push   %rbx
    2af0:	48 81 ec e8 00 00 00 	sub    $0xe8,%rsp
    2af7:	89 bd 2c ff ff ff    	mov    %edi,-0xd4(%rbp)
    2afd:	89 b5 28 ff ff ff    	mov    %esi,-0xd8(%rbp)
    2b03:	89 95 24 ff ff ff    	mov    %edx,-0xdc(%rbp)
    2b09:	48 89 8d 18 ff ff ff 	mov    %rcx,-0xe8(%rbp)
    2b10:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2b17:	00 00 
    2b19:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    2b1d:	31 c0                	xor    %eax,%eax
    2b1f:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    2b26:	8b 95 2c ff ff ff    	mov    -0xd4(%rbp),%edx
    2b2c:	89 10                	mov    %edx,(%rax)
    2b2e:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    2b35:	8b 95 28 ff ff ff    	mov    -0xd8(%rbp),%edx
    2b3b:	89 50 04             	mov    %edx,0x4(%rax)
    2b3e:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    2b45:	8b 95 24 ff ff ff    	mov    -0xdc(%rbp),%edx
    2b4b:	89 50 08             	mov    %edx,0x8(%rax)
    2b4e:	8b 85 2c ff ff ff    	mov    -0xd4(%rbp),%eax
    2b54:	48 63 d0             	movslq %eax,%rdx
    2b57:	8b 85 24 ff ff ff    	mov    -0xdc(%rbp),%eax
    2b5d:	48 98                	cltq
    2b5f:	48 0f af c2          	imul   %rdx,%rax
    2b63:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
    2b67:	8b 85 24 ff ff ff    	mov    -0xdc(%rbp),%eax
    2b6d:	48 63 d0             	movslq %eax,%rdx
    2b70:	8b 85 28 ff ff ff    	mov    -0xd8(%rbp),%eax
    2b76:	48 98                	cltq
    2b78:	48 0f af c2          	imul   %rdx,%rax
    2b7c:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
    2b80:	8b 85 2c ff ff ff    	mov    -0xd4(%rbp),%eax
    2b86:	48 63 d0             	movslq %eax,%rdx
    2b89:	8b 85 28 ff ff ff    	mov    -0xd8(%rbp),%eax
    2b8f:	48 98                	cltq
    2b91:	48 0f af c2          	imul   %rdx,%rax
    2b95:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
    2b99:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
    2b9d:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    2ba1:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
    2ba8:	48 89 d6             	mov    %rdx,%rsi
    2bab:	48 89 c7             	mov    %rax,%rdi
    2bae:	e8 26 11 00 00       	call   3cd9 <_Z10cudaMallocI6__halfE9cudaErrorPPT_m>
    2bb3:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    2bb7:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    2bbb:	48 8d 85 58 ff ff ff 	lea    -0xa8(%rbp),%rax
    2bc2:	48 89 d6             	mov    %rdx,%rsi
    2bc5:	48 89 c7             	mov    %rax,%rdi
    2bc8:	e8 0c 11 00 00       	call   3cd9 <_Z10cudaMallocI6__halfE9cudaErrorPPT_m>
    2bcd:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    2bd1:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    2bd5:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
    2bdc:	48 89 d6             	mov    %rdx,%rsi
    2bdf:	48 89 c7             	mov    %rax,%rdi
    2be2:	e8 f2 10 00 00       	call   3cd9 <_Z10cudaMallocI6__halfE9cudaErrorPPT_m>
    2be7:	48 8b 85 50 ff ff ff 	mov    -0xb0(%rbp),%rax
    2bee:	8b 95 24 ff ff ff    	mov    -0xdc(%rbp),%edx
    2bf4:	8b b5 2c ff ff ff    	mov    -0xd4(%rbp),%esi
    2bfa:	b9 00 00 00 00       	mov    $0x0,%ecx
    2bff:	48 89 c7             	mov    %rax,%rdi
    2c02:	e8 ef fd ff ff       	call   29f6 <_Z17initialize_matrixP6__halfiiy>
    2c07:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
    2c0e:	8b 95 28 ff ff ff    	mov    -0xd8(%rbp),%edx
    2c14:	8b b5 24 ff ff ff    	mov    -0xdc(%rbp),%esi
    2c1a:	b9 01 00 00 00       	mov    $0x1,%ecx
    2c1f:	48 89 c7             	mov    %rax,%rdi
    2c22:	e8 cf fd ff ff       	call   29f6 <_Z17initialize_matrixP6__halfiiy>
    2c27:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    2c2b:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    2c2f:	48 8b 85 60 ff ff ff 	mov    -0xa0(%rbp),%rax
    2c36:	be 00 00 00 00       	mov    $0x0,%esi
    2c3b:	48 89 c7             	mov    %rax,%rdi
    2c3e:	e8 4d f7 ff ff       	call   2390 <cudaMemset@plt>
    2c43:	48 8d 85 68 ff ff ff 	lea    -0x98(%rbp),%rax
    2c4a:	48 89 c7             	mov    %rax,%rdi
    2c4d:	e8 6e f9 ff ff       	call   25c0 <cublasCreate_v2@plt>
    2c52:	f3 0f 10 05 c2 37 00 	movss  0x37c2(%rip),%xmm0        # 641c <_ZSt12__is_ratio_vISt5ratioILl1000000000ELl1EEE+0x3>
    2c59:	00 
    2c5a:	f3 0f 11 85 34 ff ff 	movss  %xmm0,-0xcc(%rbp)
    2c61:	ff 
    2c62:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2c66:	f3 0f 11 85 38 ff ff 	movss  %xmm0,-0xc8(%rbp)
    2c6d:	ff 
    2c6e:	48 8b bd 60 ff ff ff 	mov    -0xa0(%rbp),%rdi
    2c75:	48 8b b5 50 ff ff ff 	mov    -0xb0(%rbp),%rsi
    2c7c:	48 8b 8d 58 ff ff ff 	mov    -0xa8(%rbp),%rcx
    2c83:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    2c8a:	44 8b 8d 24 ff ff ff 	mov    -0xdc(%rbp),%r9d
    2c91:	44 8b 95 2c ff ff ff 	mov    -0xd4(%rbp),%r10d
    2c98:	8b 95 28 ff ff ff    	mov    -0xd8(%rbp),%edx
    2c9e:	48 83 ec 08          	sub    $0x8,%rsp
    2ca2:	6a 63                	push   $0x63
    2ca4:	6a 00                	push   $0x0
    2ca6:	44 8b 85 28 ff ff ff 	mov    -0xd8(%rbp),%r8d
    2cad:	41 50                	push   %r8
    2caf:	6a 02                	push   $0x2
    2cb1:	57                   	push   %rdi
    2cb2:	48 8d bd 38 ff ff ff 	lea    -0xc8(%rbp),%rdi
    2cb9:	57                   	push   %rdi
    2cba:	8b bd 24 ff ff ff    	mov    -0xdc(%rbp),%edi
    2cc0:	57                   	push   %rdi
    2cc1:	6a 02                	push   $0x2
    2cc3:	56                   	push   %rsi
    2cc4:	8b b5 28 ff ff ff    	mov    -0xd8(%rbp),%esi
    2cca:	56                   	push   %rsi
    2ccb:	6a 02                	push   $0x2
    2ccd:	51                   	push   %rcx
    2cce:	48 8d 8d 34 ff ff ff 	lea    -0xcc(%rbp),%rcx
    2cd5:	51                   	push   %rcx
    2cd6:	45 89 d0             	mov    %r10d,%r8d
    2cd9:	89 d1                	mov    %edx,%ecx
    2cdb:	ba 00 00 00 00       	mov    $0x0,%edx
    2ce0:	be 00 00 00 00       	mov    $0x0,%esi
    2ce5:	48 89 c7             	mov    %rax,%rdi
    2ce8:	e8 bf fb ff ff       	call   28ac <_ZL12cublasGemmExP13cublasContext17cublasOperation_tS1_iiiPKvS3_14cudaDataType_tiS3_S4_iS3_PvS4_iS4_16cublasGemmAlgo_t>
    2ced:	48 83 c4 70          	add    $0x70,%rsp
    2cf1:	e8 0a f9 ff ff       	call   2600 <cudaDeviceSynchronize@plt>
    2cf6:	c7 85 44 ff ff ff 0a 	movl   $0xa,-0xbc(%rbp)
    2cfd:	00 00 00 
    2d00:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
    2d04:	48 89 c7             	mov    %rax,%rdi
    2d07:	e8 68 12 00 00       	call   3f74 <_ZNSt6vectorIfSaIfEEC1Ev>
    2d0c:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    2d13:	48 89 c7             	mov    %rax,%rdi
    2d16:	e8 95 f6 ff ff       	call   23b0 <cudaEventCreate@plt>
    2d1b:	48 8d 85 78 ff ff ff 	lea    -0x88(%rbp),%rax
    2d22:	48 89 c7             	mov    %rax,%rdi
    2d25:	e8 86 f6 ff ff       	call   23b0 <cudaEventCreate@plt>
    2d2a:	c7 85 3c ff ff ff 00 	movl   $0x0,-0xc4(%rbp)
    2d31:	00 00 00 
    2d34:	e9 f4 00 00 00       	jmp    2e2d <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x346>
    2d39:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
    2d40:	be 00 00 00 00       	mov    $0x0,%esi
    2d45:	48 89 c7             	mov    %rax,%rdi
    2d48:	e8 f3 f6 ff ff       	call   2440 <cudaEventRecord@plt>
    2d4d:	48 8b bd 60 ff ff ff 	mov    -0xa0(%rbp),%rdi
    2d54:	48 8b b5 50 ff ff ff 	mov    -0xb0(%rbp),%rsi
    2d5b:	48 8b 8d 58 ff ff ff 	mov    -0xa8(%rbp),%rcx
    2d62:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    2d69:	44 8b 8d 24 ff ff ff 	mov    -0xdc(%rbp),%r9d
    2d70:	44 8b 95 2c ff ff ff 	mov    -0xd4(%rbp),%r10d
    2d77:	8b 95 28 ff ff ff    	mov    -0xd8(%rbp),%edx
    2d7d:	48 83 ec 08          	sub    $0x8,%rsp
    2d81:	6a 63                	push   $0x63
    2d83:	6a 00                	push   $0x0
    2d85:	44 8b 85 28 ff ff ff 	mov    -0xd8(%rbp),%r8d
    2d8c:	41 50                	push   %r8
    2d8e:	6a 02                	push   $0x2
    2d90:	57                   	push   %rdi
    2d91:	48 8d bd 38 ff ff ff 	lea    -0xc8(%rbp),%rdi
    2d98:	57                   	push   %rdi
    2d99:	8b bd 24 ff ff ff    	mov    -0xdc(%rbp),%edi
    2d9f:	57                   	push   %rdi
    2da0:	6a 02                	push   $0x2
    2da2:	56                   	push   %rsi
    2da3:	8b b5 28 ff ff ff    	mov    -0xd8(%rbp),%esi
    2da9:	56                   	push   %rsi
    2daa:	6a 02                	push   $0x2
    2dac:	51                   	push   %rcx
    2dad:	48 8d 8d 34 ff ff ff 	lea    -0xcc(%rbp),%rcx
    2db4:	51                   	push   %rcx
    2db5:	45 89 d0             	mov    %r10d,%r8d
    2db8:	89 d1                	mov    %edx,%ecx
    2dba:	ba 00 00 00 00       	mov    $0x0,%edx
    2dbf:	be 00 00 00 00       	mov    $0x0,%esi
    2dc4:	48 89 c7             	mov    %rax,%rdi
    2dc7:	e8 e0 fa ff ff       	call   28ac <_ZL12cublasGemmExP13cublasContext17cublasOperation_tS1_iiiPKvS3_14cudaDataType_tiS3_S4_iS3_PvS4_iS4_16cublasGemmAlgo_t>
    2dcc:	48 83 c4 70          	add    $0x70,%rsp
    2dd0:	48 8b 85 78 ff ff ff 	mov    -0x88(%rbp),%rax
    2dd7:	be 00 00 00 00       	mov    $0x0,%esi
    2ddc:	48 89 c7             	mov    %rax,%rdi
    2ddf:	e8 5c f6 ff ff       	call   2440 <cudaEventRecord@plt>
    2de4:	48 8b 85 78 ff ff ff 	mov    -0x88(%rbp),%rax
    2deb:	48 89 c7             	mov    %rax,%rdi
    2dee:	e8 dd f7 ff ff       	call   25d0 <cudaEventSynchronize@plt>
    2df3:	48 8b 95 78 ff ff ff 	mov    -0x88(%rbp),%rdx
    2dfa:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
    2e01:	48 89 d6             	mov    %rdx,%rsi
    2e04:	48 89 c7             	mov    %rax,%rdi
    2e07:	e8 95 fb ff ff       	call   29a1 <_Z10elapsed_msP10CUevent_stS0_>
    2e0c:	66 0f 7e c0          	movd   %xmm0,%eax
    2e10:	89 45 88             	mov    %eax,-0x78(%rbp)
    2e13:	48 8d 55 88          	lea    -0x78(%rbp),%rdx
    2e17:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
    2e1b:	48 89 d6             	mov    %rdx,%rsi
    2e1e:	48 89 c7             	mov    %rax,%rdi
    2e21:	e8 54 13 00 00       	call   417a <_ZNSt6vectorIfSaIfEE9push_backEOf>
    2e26:	83 85 3c ff ff ff 01 	addl   $0x1,-0xc4(%rbp)
    2e2d:	8b 85 3c ff ff ff    	mov    -0xc4(%rbp),%eax
    2e33:	3b 85 44 ff ff ff    	cmp    -0xbc(%rbp),%eax
    2e39:	0f 8c fa fe ff ff    	jl     2d39 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x252>
    2e3f:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2e43:	f3 0f 11 85 40 ff ff 	movss  %xmm0,-0xc0(%rbp)
    2e4a:	ff 
    2e4b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
    2e4f:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
    2e53:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
    2e57:	48 89 c7             	mov    %rax,%rdi
    2e5a:	e8 51 13 00 00       	call   41b0 <_ZNSt6vectorIfSaIfEE5beginEv>
    2e5f:	48 89 45 80          	mov    %rax,-0x80(%rbp)
    2e63:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
    2e67:	48 89 c7             	mov    %rax,%rdi
    2e6a:	e8 8d 13 00 00       	call   41fc <_ZNSt6vectorIfSaIfEE3endEv>
    2e6f:	48 89 45 88          	mov    %rax,-0x78(%rbp)
    2e73:	eb 3c                	jmp    2eb1 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x3ca>
    2e75:	48 8d 45 80          	lea    -0x80(%rbp),%rax
    2e79:	48 89 c7             	mov    %rax,%rdi
    2e7c:	e8 2f 14 00 00       	call   42b0 <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEdeEv>
    2e81:	f3 0f 10 00          	movss  (%rax),%xmm0
    2e85:	f3 0f 11 85 4c ff ff 	movss  %xmm0,-0xb4(%rbp)
    2e8c:	ff 
    2e8d:	f3 0f 10 85 40 ff ff 	movss  -0xc0(%rbp),%xmm0
    2e94:	ff 
    2e95:	f3 0f 58 85 4c ff ff 	addss  -0xb4(%rbp),%xmm0
    2e9c:	ff 
    2e9d:	f3 0f 11 85 40 ff ff 	movss  %xmm0,-0xc0(%rbp)
    2ea4:	ff 
    2ea5:	48 8d 45 80          	lea    -0x80(%rbp),%rax
    2ea9:	48 89 c7             	mov    %rax,%rdi
    2eac:	e8 db 13 00 00       	call   428c <_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEppEv>
    2eb1:	48 8d 55 88          	lea    -0x78(%rbp),%rdx
    2eb5:	48 8d 45 80          	lea    -0x80(%rbp),%rax
    2eb9:	48 89 d6             	mov    %rdx,%rsi
    2ebc:	48 89 c7             	mov    %rax,%rdi
    2ebf:	e8 88 13 00 00       	call   424c <_ZN9__gnu_cxxneIPfSt6vectorIfSaIfEEEEbRKNS_17__normal_iteratorIT_T0_EESA_>
    2ec4:	84 c0                	test   %al,%al
    2ec6:	75 ad                	jne    2e75 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x38e>
    2ec8:	66 0f ef c9          	pxor   %xmm1,%xmm1
    2ecc:	f3 0f 2a 8d 44 ff ff 	cvtsi2ssl -0xbc(%rbp),%xmm1
    2ed3:	ff 
    2ed4:	f3 0f 10 85 40 ff ff 	movss  -0xc0(%rbp),%xmm0
    2edb:	ff 
    2edc:	f3 0f 5e c1          	divss  %xmm1,%xmm0
    2ee0:	f3 0f 11 85 48 ff ff 	movss  %xmm0,-0xb8(%rbp)
    2ee7:	ff 
    2ee8:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2eec:	f2 0f 2a 85 2c ff ff 	cvtsi2sdl -0xd4(%rbp),%xmm0
    2ef3:	ff 
    2ef4:	66 0f 28 c8          	movapd %xmm0,%xmm1
    2ef8:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
    2efc:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2f00:	f2 0f 2a 85 28 ff ff 	cvtsi2sdl -0xd8(%rbp),%xmm0
    2f07:	ff 
    2f08:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
    2f0c:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2f10:	f2 0f 2a 85 24 ff ff 	cvtsi2sdl -0xdc(%rbp),%xmm0
    2f17:	ff 
    2f18:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    2f1c:	66 0f ef d2          	pxor   %xmm2,%xmm2
    2f20:	f3 0f 5a 95 48 ff ff 	cvtss2sd -0xb8(%rbp),%xmm2
    2f27:	ff 
    2f28:	f2 0f 10 0d f8 34 00 	movsd  0x34f8(%rip),%xmm1        # 6428 <_ZSt12__is_ratio_vISt5ratioILl1000000000ELl1EEE+0xf>
    2f2f:	00 
    2f30:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
    2f34:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
    2f38:	f2 0f 11 45 c0       	movsd  %xmm0,-0x40(%rbp)
    2f3d:	48 8b 5d b0          	mov    -0x50(%rbp),%rbx
    2f41:	48 b8 fc ff ff ff ff 	movabs $0x3ffffffffffffffc,%rax
    2f48:	ff ff 3f 
    2f4b:	48 39 d8             	cmp    %rbx,%rax
    2f4e:	72 0e                	jb     2f5e <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x477>
    2f50:	48 8d 04 1b          	lea    (%rbx,%rbx,1),%rax
    2f54:	48 89 c7             	mov    %rax,%rdi
    2f57:	e8 04 f4 ff ff       	call   2360 <_Znam@plt>
    2f5c:	eb 05                	jmp    2f63 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x47c>
    2f5e:	e8 2d f6 ff ff       	call   2590 <__cxa_throw_bad_array_new_length@plt>
    2f63:	48 89 c2             	mov    %rax,%rdx
    2f66:	48 8d 43 ff          	lea    -0x1(%rbx),%rax
    2f6a:	eb 04                	jmp    2f70 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x489>
    2f6c:	48 83 e8 01          	sub    $0x1,%rax
    2f70:	48 85 c0             	test   %rax,%rax
    2f73:	79 f7                	jns    2f6c <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x485>
    2f75:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
    2f79:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    2f7d:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    2f81:	48 8b b5 60 ff ff ff 	mov    -0xa0(%rbp),%rsi
    2f88:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    2f8c:	b9 02 00 00 00       	mov    $0x2,%ecx
    2f91:	48 89 c7             	mov    %rax,%rdi
    2f94:	e8 b7 f5 ff ff       	call   2550 <cudaMemcpy@plt>
    2f99:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2f9d:	f2 0f 11 45 90       	movsd  %xmm0,-0x70(%rbp)
    2fa2:	48 c7 45 98 00 00 00 	movq   $0x0,-0x68(%rbp)
    2fa9:	00 
    2faa:	eb 2e                	jmp    2fda <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x4f3>
    2fac:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    2fb0:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    2fb4:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    2fb8:	48 01 d0             	add    %rdx,%rax
    2fbb:	48 89 c7             	mov    %rax,%rdi
    2fbe:	e8 05 0e 00 00       	call   3dc8 <_ZNK6__halfcvfEv>
    2fc3:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
    2fc7:	f2 0f 10 4d 90       	movsd  -0x70(%rbp),%xmm1
    2fcc:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    2fd0:	f2 0f 11 45 90       	movsd  %xmm0,-0x70(%rbp)
    2fd5:	48 83 45 98 01       	addq   $0x1,-0x68(%rbp)
    2fda:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    2fde:	48 3b 45 b0          	cmp    -0x50(%rbp),%rax
    2fe2:	72 c8                	jb     2fac <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x4c5>
    2fe4:	48 83 7d c8 00       	cmpq   $0x0,-0x38(%rbp)
    2fe9:	74 0c                	je     2ff7 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x510>
    2feb:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    2fef:	48 89 c7             	mov    %rax,%rdi
    2ff2:	e8 89 f5 ff ff       	call   2580 <_ZdaPv@plt>
    2ff7:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    2ffe:	f3 0f 10 85 48 ff ff 	movss  -0xb8(%rbp),%xmm0
    3005:	ff 
    3006:	f3 0f 11 40 0c       	movss  %xmm0,0xc(%rax)
    300b:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    3012:	f2 0f 10 45 c0       	movsd  -0x40(%rbp),%xmm0
    3017:	f2 0f 11 40 10       	movsd  %xmm0,0x10(%rax)
    301c:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    3023:	f2 0f 10 45 90       	movsd  -0x70(%rbp),%xmm0
    3028:	f2 0f 11 40 18       	movsd  %xmm0,0x18(%rax)
    302d:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    3034:	c6 40 20 01          	movb   $0x1,0x20(%rax)
    3038:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    303f:	48 89 c7             	mov    %rax,%rdi
    3042:	e8 19 f5 ff ff       	call   2560 <cublasDestroy_v2@plt>
    3047:	48 8b 85 50 ff ff ff 	mov    -0xb0(%rbp),%rax
    304e:	48 89 c7             	mov    %rax,%rdi
    3051:	e8 4a f5 ff ff       	call   25a0 <cudaFree@plt>
    3056:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
    305d:	48 89 c7             	mov    %rax,%rdi
    3060:	e8 3b f5 ff ff       	call   25a0 <cudaFree@plt>
    3065:	48 8b 85 60 ff ff ff 	mov    -0xa0(%rbp),%rax
    306c:	48 89 c7             	mov    %rax,%rdi
    306f:	e8 2c f5 ff ff       	call   25a0 <cudaFree@plt>
    3074:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
    307b:	48 89 c7             	mov    %rax,%rdi
    307e:	e8 9d f3 ff ff       	call   2420 <cudaEventDestroy@plt>
    3083:	48 8b 85 78 ff ff ff 	mov    -0x88(%rbp),%rax
    308a:	48 89 c7             	mov    %rax,%rdi
    308d:	e8 8e f3 ff ff       	call   2420 <cudaEventDestroy@plt>
    3092:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
    3096:	48 89 c7             	mov    %rax,%rdi
    3099:	e8 94 10 00 00       	call   4132 <_ZNSt6vectorIfSaIfEED1Ev>
    309e:	eb 64                	jmp    3104 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x61d>
    30a0:	f3 0f 1e fa          	endbr64
    30a4:	48 89 c3             	mov    %rax,%rbx
    30a7:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
    30ab:	48 89 c7             	mov    %rax,%rdi
    30ae:	e8 7f 10 00 00       	call   4132 <_ZNSt6vectorIfSaIfEED1Ev>
    30b3:	48 89 d8             	mov    %rbx,%rax
    30b6:	eb 04                	jmp    30bc <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x5d5>
    30b8:	f3 0f 1e fa          	endbr64
    30bc:	48 89 c7             	mov    %rax,%rdi
    30bf:	e8 2c f3 ff ff       	call   23f0 <__cxa_begin_catch@plt>
    30c4:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    30cb:	c6 40 20 00          	movb   $0x0,0x20(%rax)
    30cf:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    30d6:	66 0f ef c0          	pxor   %xmm0,%xmm0
    30da:	f3 0f 11 40 0c       	movss  %xmm0,0xc(%rax)
    30df:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    30e6:	66 0f ef c0          	pxor   %xmm0,%xmm0
    30ea:	f2 0f 11 40 10       	movsd  %xmm0,0x10(%rax)
    30ef:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    30f6:	66 0f ef c0          	pxor   %xmm0,%xmm0
    30fa:	f2 0f 11 40 18       	movsd  %xmm0,0x18(%rax)
    30ff:	e8 ec f4 ff ff       	call   25f0 <__cxa_end_catch@plt>
    3104:	90                   	nop
    3105:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    3109:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    3110:	00 00 
    3112:	74 05                	je     3119 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult+0x632>
    3114:	e8 17 f4 ff ff       	call   2530 <__stack_chk_fail@plt>
    3119:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    311d:	c9                   	leave
    311e:	c3                   	ret

000000000000311f <main>:
    311f:	f3 0f 1e fa          	endbr64
    3123:	55                   	push   %rbp
    3124:	48 89 e5             	mov    %rsp,%rbp
    3127:	53                   	push   %rbx
    3128:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
    312f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    3136:	00 00 
    3138:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    313c:	31 c0                	xor    %eax,%eax
    313e:	48 8d 05 23 30 00 00 	lea    0x3023(%rip),%rax        # 6168 <_ZN2nv6targetL5sm_90E+0x38>
    3145:	48 89 c6             	mov    %rax,%rsi
    3148:	48 8d 05 f1 5e 00 00 	lea    0x5ef1(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    314f:	48 89 c7             	mov    %rax,%rdi
    3152:	e8 69 f3 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3157:	48 8b 15 72 5e 00 00 	mov    0x5e72(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    315e:	48 89 d6             	mov    %rdx,%rsi
    3161:	48 89 c7             	mov    %rax,%rdi
    3164:	e8 97 f3 ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3169:	48 8d 45 90          	lea    -0x70(%rbp),%rax
    316d:	48 89 c7             	mov    %rax,%rdi
    3170:	e8 cb f4 ff ff       	call   2640 <_ZNSaIcEC1Ev@plt>
    3175:	48 8d 55 90          	lea    -0x70(%rbp),%rdx
    3179:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
    317d:	48 89 d1             	mov    %rdx,%rcx
    3180:	ba 3d 00 00 00       	mov    $0x3d,%edx
    3185:	be 32 00 00 00       	mov    $0x32,%esi
    318a:	48 89 c7             	mov    %rax,%rdi
    318d:	e8 82 0e 00 00       	call   4014 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEmcRKS3_>
    3192:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
    3196:	48 89 c6             	mov    %rax,%rsi
    3199:	48 8d 05 a0 5e 00 00 	lea    0x5ea0(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    31a0:	48 89 c7             	mov    %rax,%rdi
    31a3:	e8 f8 f2 ff ff       	call   24a0 <_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@plt>
    31a8:	48 8b 15 21 5e 00 00 	mov    0x5e21(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    31af:	48 89 d6             	mov    %rdx,%rsi
    31b2:	48 89 c7             	mov    %rax,%rdi
    31b5:	e8 46 f3 ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    31ba:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
    31be:	48 89 c7             	mov    %rax,%rdi
    31c1:	e8 6a f2 ff ff       	call   2430 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@plt>
    31c6:	48 8d 45 90          	lea    -0x70(%rbp),%rax
    31ca:	48 89 c7             	mov    %rax,%rdi
    31cd:	e8 4e f3 ff ff       	call   2520 <_ZNSaIcED1Ev@plt>
    31d2:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    31d9:	48 89 c7             	mov    %rax,%rdi
    31dc:	e8 f3 0d 00 00       	call   3fd4 <_ZNSt6vectorI15BenchmarkResultSaIS0_EEC1Ev>
    31e1:	c7 85 48 ff ff ff 00 	movl   $0x400,-0xb8(%rbp)
    31e8:	04 00 00 
    31eb:	c7 85 4c ff ff ff 00 	movl   $0x200,-0xb4(%rbp)
    31f2:	02 00 00 
    31f5:	c7 85 50 ff ff ff 00 	movl   $0x400,-0xb0(%rbp)
    31fc:	04 00 00 
    31ff:	48 8d 4d 90          	lea    -0x70(%rbp),%rcx
    3203:	8b 95 50 ff ff ff    	mov    -0xb0(%rbp),%edx
    3209:	8b b5 4c ff ff ff    	mov    -0xb4(%rbp),%esi
    320f:	8b 85 48 ff ff ff    	mov    -0xb8(%rbp),%eax
    3215:	89 c7                	mov    %eax,%edi
    3217:	e8 cb f8 ff ff       	call   2ae7 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult>
    321c:	48 8d 55 90          	lea    -0x70(%rbp),%rdx
    3220:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    3227:	48 89 d6             	mov    %rdx,%rsi
    322a:	48 89 c7             	mov    %rax,%rdi
    322d:	e8 84 11 00 00       	call   43b6 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE9push_backERKS0_>
    3232:	0f b6 45 b0          	movzbl -0x50(%rbp),%eax
    3236:	84 c0                	test   %al,%al
    3238:	0f 84 2e 01 00 00    	je     336c <main+0x24d>
    323e:	48 8d 05 44 2f 00 00 	lea    0x2f44(%rip),%rax        # 6189 <_ZN2nv6targetL5sm_90E+0x59>
    3245:	48 89 c6             	mov    %rax,%rsi
    3248:	48 8d 05 f1 5d 00 00 	lea    0x5df1(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    324f:	48 89 c7             	mov    %rax,%rdi
    3252:	e8 69 f2 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3257:	48 89 c2             	mov    %rax,%rdx
    325a:	8b 45 90             	mov    -0x70(%rbp),%eax
    325d:	89 c6                	mov    %eax,%esi
    325f:	48 89 d7             	mov    %rdx,%rdi
    3262:	e8 b9 f3 ff ff       	call   2620 <_ZNSolsEi@plt>
    3267:	48 89 c2             	mov    %rax,%rdx
    326a:	48 8d 05 26 2f 00 00 	lea    0x2f26(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    3271:	48 89 c6             	mov    %rax,%rsi
    3274:	48 89 d7             	mov    %rdx,%rdi
    3277:	e8 44 f2 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    327c:	48 89 c2             	mov    %rax,%rdx
    327f:	8b 45 94             	mov    -0x6c(%rbp),%eax
    3282:	89 c6                	mov    %eax,%esi
    3284:	48 89 d7             	mov    %rdx,%rdi
    3287:	e8 94 f3 ff ff       	call   2620 <_ZNSolsEi@plt>
    328c:	48 89 c2             	mov    %rax,%rdx
    328f:	48 8d 05 01 2f 00 00 	lea    0x2f01(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    3296:	48 89 c6             	mov    %rax,%rsi
    3299:	48 89 d7             	mov    %rdx,%rdi
    329c:	e8 1f f2 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    32a1:	48 89 c2             	mov    %rax,%rdx
    32a4:	8b 45 98             	mov    -0x68(%rbp),%eax
    32a7:	89 c6                	mov    %eax,%esi
    32a9:	48 89 d7             	mov    %rdx,%rdi
    32ac:	e8 6f f3 ff ff       	call   2620 <_ZNSolsEi@plt>
    32b1:	48 89 c2             	mov    %rax,%rdx
    32b4:	48 8d 05 de 2e 00 00 	lea    0x2ede(%rip),%rax        # 6199 <_ZN2nv6targetL5sm_90E+0x69>
    32bb:	48 89 c6             	mov    %rax,%rsi
    32be:	48 89 d7             	mov    %rdx,%rdi
    32c1:	e8 fa f1 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    32c6:	48 89 c2             	mov    %rax,%rdx
    32c9:	8b 45 9c             	mov    -0x64(%rbp),%eax
    32cc:	66 0f 6e c0          	movd   %eax,%xmm0
    32d0:	48 89 d7             	mov    %rdx,%rdi
    32d3:	e8 78 f1 ff ff       	call   2450 <_ZNSolsEf@plt>
    32d8:	48 89 c2             	mov    %rax,%rdx
    32db:	48 8d 05 bb 2e 00 00 	lea    0x2ebb(%rip),%rax        # 619d <_ZN2nv6targetL5sm_90E+0x6d>
    32e2:	48 89 c6             	mov    %rax,%rsi
    32e5:	48 89 d7             	mov    %rdx,%rdi
    32e8:	e8 d3 f1 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    32ed:	48 89 c2             	mov    %rax,%rdx
    32f0:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
    32f4:	66 48 0f 6e c0       	movq   %rax,%xmm0
    32f9:	48 89 d7             	mov    %rdx,%rdi
    32fc:	e8 6f f3 ff ff       	call   2670 <_ZNSolsEd@plt>
    3301:	48 89 c2             	mov    %rax,%rdx
    3304:	48 8d 05 a9 2e 00 00 	lea    0x2ea9(%rip),%rax        # 61b4 <_ZN2nv6targetL5sm_90E+0x84>
    330b:	48 89 c6             	mov    %rax,%rsi
    330e:	48 89 d7             	mov    %rdx,%rdi
    3311:	e8 aa f1 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3316:	48 8b 15 b3 5c 00 00 	mov    0x5cb3(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    331d:	48 89 d6             	mov    %rdx,%rsi
    3320:	48 89 c7             	mov    %rax,%rdi
    3323:	e8 d8 f1 ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3328:	48 8d 05 8d 2e 00 00 	lea    0x2e8d(%rip),%rax        # 61bc <_ZN2nv6targetL5sm_90E+0x8c>
    332f:	48 89 c6             	mov    %rax,%rsi
    3332:	48 8d 05 07 5d 00 00 	lea    0x5d07(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3339:	48 89 c7             	mov    %rax,%rdi
    333c:	e8 7f f1 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3341:	48 89 c2             	mov    %rax,%rdx
    3344:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    3348:	66 48 0f 6e c0       	movq   %rax,%xmm0
    334d:	48 89 d7             	mov    %rdx,%rdi
    3350:	e8 1b f3 ff ff       	call   2670 <_ZNSolsEd@plt>
    3355:	48 8b 15 74 5c 00 00 	mov    0x5c74(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    335c:	48 89 d6             	mov    %rdx,%rsi
    335f:	48 89 c7             	mov    %rax,%rdi
    3362:	e8 99 f1 ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3367:	e9 a3 00 00 00       	jmp    340f <main+0x2f0>
    336c:	48 8d 05 16 2e 00 00 	lea    0x2e16(%rip),%rax        # 6189 <_ZN2nv6targetL5sm_90E+0x59>
    3373:	48 89 c6             	mov    %rax,%rsi
    3376:	48 8d 05 c3 5c 00 00 	lea    0x5cc3(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    337d:	48 89 c7             	mov    %rax,%rdi
    3380:	e8 3b f1 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3385:	48 89 c2             	mov    %rax,%rdx
    3388:	8b 85 48 ff ff ff    	mov    -0xb8(%rbp),%eax
    338e:	89 c6                	mov    %eax,%esi
    3390:	48 89 d7             	mov    %rdx,%rdi
    3393:	e8 88 f2 ff ff       	call   2620 <_ZNSolsEi@plt>
    3398:	48 89 c2             	mov    %rax,%rdx
    339b:	48 8d 05 f5 2d 00 00 	lea    0x2df5(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    33a2:	48 89 c6             	mov    %rax,%rsi
    33a5:	48 89 d7             	mov    %rdx,%rdi
    33a8:	e8 13 f1 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    33ad:	48 89 c2             	mov    %rax,%rdx
    33b0:	8b 85 4c ff ff ff    	mov    -0xb4(%rbp),%eax
    33b6:	89 c6                	mov    %eax,%esi
    33b8:	48 89 d7             	mov    %rdx,%rdi
    33bb:	e8 60 f2 ff ff       	call   2620 <_ZNSolsEi@plt>
    33c0:	48 89 c2             	mov    %rax,%rdx
    33c3:	48 8d 05 cd 2d 00 00 	lea    0x2dcd(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    33ca:	48 89 c6             	mov    %rax,%rsi
    33cd:	48 89 d7             	mov    %rdx,%rdi
    33d0:	e8 eb f0 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    33d5:	48 89 c2             	mov    %rax,%rdx
    33d8:	8b 85 50 ff ff ff    	mov    -0xb0(%rbp),%eax
    33de:	89 c6                	mov    %eax,%esi
    33e0:	48 89 d7             	mov    %rdx,%rdi
    33e3:	e8 38 f2 ff ff       	call   2620 <_ZNSolsEi@plt>
    33e8:	48 89 c2             	mov    %rax,%rdx
    33eb:	48 8d 05 dc 2d 00 00 	lea    0x2ddc(%rip),%rax        # 61ce <_ZN2nv6targetL5sm_90E+0x9e>
    33f2:	48 89 c6             	mov    %rax,%rsi
    33f5:	48 89 d7             	mov    %rdx,%rdi
    33f8:	e8 c3 f0 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    33fd:	48 8b 15 cc 5b 00 00 	mov    0x5bcc(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3404:	48 89 d6             	mov    %rdx,%rsi
    3407:	48 89 c7             	mov    %rax,%rdi
    340a:	e8 f1 f0 ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    340f:	c7 85 54 ff ff ff 00 	movl   $0x1000,-0xac(%rbp)
    3416:	10 00 00 
    3419:	c7 85 58 ff ff ff 00 	movl   $0x800,-0xa8(%rbp)
    3420:	08 00 00 
    3423:	c7 85 5c ff ff ff 00 	movl   $0x1000,-0xa4(%rbp)
    342a:	10 00 00 
    342d:	48 8d 4d 90          	lea    -0x70(%rbp),%rcx
    3431:	8b 95 5c ff ff ff    	mov    -0xa4(%rbp),%edx
    3437:	8b b5 58 ff ff ff    	mov    -0xa8(%rbp),%esi
    343d:	8b 85 54 ff ff ff    	mov    -0xac(%rbp),%eax
    3443:	89 c7                	mov    %eax,%edi
    3445:	e8 9d f6 ff ff       	call   2ae7 <_Z27run_cublas_benchmark_singleiiiR15BenchmarkResult>
    344a:	48 8d 55 90          	lea    -0x70(%rbp),%rdx
    344e:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    3455:	48 89 d6             	mov    %rdx,%rsi
    3458:	48 89 c7             	mov    %rax,%rdi
    345b:	e8 56 0f 00 00       	call   43b6 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE9push_backERKS0_>
    3460:	0f b6 45 b0          	movzbl -0x50(%rbp),%eax
    3464:	84 c0                	test   %al,%al
    3466:	0f 84 2e 01 00 00    	je     359a <main+0x47b>
    346c:	48 8d 05 16 2d 00 00 	lea    0x2d16(%rip),%rax        # 6189 <_ZN2nv6targetL5sm_90E+0x59>
    3473:	48 89 c6             	mov    %rax,%rsi
    3476:	48 8d 05 c3 5b 00 00 	lea    0x5bc3(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    347d:	48 89 c7             	mov    %rax,%rdi
    3480:	e8 3b f0 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3485:	48 89 c2             	mov    %rax,%rdx
    3488:	8b 45 90             	mov    -0x70(%rbp),%eax
    348b:	89 c6                	mov    %eax,%esi
    348d:	48 89 d7             	mov    %rdx,%rdi
    3490:	e8 8b f1 ff ff       	call   2620 <_ZNSolsEi@plt>
    3495:	48 89 c2             	mov    %rax,%rdx
    3498:	48 8d 05 f8 2c 00 00 	lea    0x2cf8(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    349f:	48 89 c6             	mov    %rax,%rsi
    34a2:	48 89 d7             	mov    %rdx,%rdi
    34a5:	e8 16 f0 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    34aa:	48 89 c2             	mov    %rax,%rdx
    34ad:	8b 45 94             	mov    -0x6c(%rbp),%eax
    34b0:	89 c6                	mov    %eax,%esi
    34b2:	48 89 d7             	mov    %rdx,%rdi
    34b5:	e8 66 f1 ff ff       	call   2620 <_ZNSolsEi@plt>
    34ba:	48 89 c2             	mov    %rax,%rdx
    34bd:	48 8d 05 d3 2c 00 00 	lea    0x2cd3(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    34c4:	48 89 c6             	mov    %rax,%rsi
    34c7:	48 89 d7             	mov    %rdx,%rdi
    34ca:	e8 f1 ef ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    34cf:	48 89 c2             	mov    %rax,%rdx
    34d2:	8b 45 98             	mov    -0x68(%rbp),%eax
    34d5:	89 c6                	mov    %eax,%esi
    34d7:	48 89 d7             	mov    %rdx,%rdi
    34da:	e8 41 f1 ff ff       	call   2620 <_ZNSolsEi@plt>
    34df:	48 89 c2             	mov    %rax,%rdx
    34e2:	48 8d 05 b0 2c 00 00 	lea    0x2cb0(%rip),%rax        # 6199 <_ZN2nv6targetL5sm_90E+0x69>
    34e9:	48 89 c6             	mov    %rax,%rsi
    34ec:	48 89 d7             	mov    %rdx,%rdi
    34ef:	e8 cc ef ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    34f4:	48 89 c2             	mov    %rax,%rdx
    34f7:	8b 45 9c             	mov    -0x64(%rbp),%eax
    34fa:	66 0f 6e c0          	movd   %eax,%xmm0
    34fe:	48 89 d7             	mov    %rdx,%rdi
    3501:	e8 4a ef ff ff       	call   2450 <_ZNSolsEf@plt>
    3506:	48 89 c2             	mov    %rax,%rdx
    3509:	48 8d 05 8d 2c 00 00 	lea    0x2c8d(%rip),%rax        # 619d <_ZN2nv6targetL5sm_90E+0x6d>
    3510:	48 89 c6             	mov    %rax,%rsi
    3513:	48 89 d7             	mov    %rdx,%rdi
    3516:	e8 a5 ef ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    351b:	48 89 c2             	mov    %rax,%rdx
    351e:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
    3522:	66 48 0f 6e c0       	movq   %rax,%xmm0
    3527:	48 89 d7             	mov    %rdx,%rdi
    352a:	e8 41 f1 ff ff       	call   2670 <_ZNSolsEd@plt>
    352f:	48 89 c2             	mov    %rax,%rdx
    3532:	48 8d 05 7b 2c 00 00 	lea    0x2c7b(%rip),%rax        # 61b4 <_ZN2nv6targetL5sm_90E+0x84>
    3539:	48 89 c6             	mov    %rax,%rsi
    353c:	48 89 d7             	mov    %rdx,%rdi
    353f:	e8 7c ef ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3544:	48 8b 15 85 5a 00 00 	mov    0x5a85(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    354b:	48 89 d6             	mov    %rdx,%rsi
    354e:	48 89 c7             	mov    %rax,%rdi
    3551:	e8 aa ef ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3556:	48 8d 05 5f 2c 00 00 	lea    0x2c5f(%rip),%rax        # 61bc <_ZN2nv6targetL5sm_90E+0x8c>
    355d:	48 89 c6             	mov    %rax,%rsi
    3560:	48 8d 05 d9 5a 00 00 	lea    0x5ad9(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3567:	48 89 c7             	mov    %rax,%rdi
    356a:	e8 51 ef ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    356f:	48 89 c2             	mov    %rax,%rdx
    3572:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    3576:	66 48 0f 6e c0       	movq   %rax,%xmm0
    357b:	48 89 d7             	mov    %rdx,%rdi
    357e:	e8 ed f0 ff ff       	call   2670 <_ZNSolsEd@plt>
    3583:	48 8b 15 46 5a 00 00 	mov    0x5a46(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    358a:	48 89 d6             	mov    %rdx,%rsi
    358d:	48 89 c7             	mov    %rax,%rdi
    3590:	e8 6b ef ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3595:	e9 a3 00 00 00       	jmp    363d <main+0x51e>
    359a:	48 8d 05 e8 2b 00 00 	lea    0x2be8(%rip),%rax        # 6189 <_ZN2nv6targetL5sm_90E+0x59>
    35a1:	48 89 c6             	mov    %rax,%rsi
    35a4:	48 8d 05 95 5a 00 00 	lea    0x5a95(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    35ab:	48 89 c7             	mov    %rax,%rdi
    35ae:	e8 0d ef ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    35b3:	48 89 c2             	mov    %rax,%rdx
    35b6:	8b 85 54 ff ff ff    	mov    -0xac(%rbp),%eax
    35bc:	89 c6                	mov    %eax,%esi
    35be:	48 89 d7             	mov    %rdx,%rdi
    35c1:	e8 5a f0 ff ff       	call   2620 <_ZNSolsEi@plt>
    35c6:	48 89 c2             	mov    %rax,%rdx
    35c9:	48 8d 05 c7 2b 00 00 	lea    0x2bc7(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    35d0:	48 89 c6             	mov    %rax,%rsi
    35d3:	48 89 d7             	mov    %rdx,%rdi
    35d6:	e8 e5 ee ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    35db:	48 89 c2             	mov    %rax,%rdx
    35de:	8b 85 58 ff ff ff    	mov    -0xa8(%rbp),%eax
    35e4:	89 c6                	mov    %eax,%esi
    35e6:	48 89 d7             	mov    %rdx,%rdi
    35e9:	e8 32 f0 ff ff       	call   2620 <_ZNSolsEi@plt>
    35ee:	48 89 c2             	mov    %rax,%rdx
    35f1:	48 8d 05 9f 2b 00 00 	lea    0x2b9f(%rip),%rax        # 6197 <_ZN2nv6targetL5sm_90E+0x67>
    35f8:	48 89 c6             	mov    %rax,%rsi
    35fb:	48 89 d7             	mov    %rdx,%rdi
    35fe:	e8 bd ee ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3603:	48 89 c2             	mov    %rax,%rdx
    3606:	8b 85 5c ff ff ff    	mov    -0xa4(%rbp),%eax
    360c:	89 c6                	mov    %eax,%esi
    360e:	48 89 d7             	mov    %rdx,%rdi
    3611:	e8 0a f0 ff ff       	call   2620 <_ZNSolsEi@plt>
    3616:	48 89 c2             	mov    %rax,%rdx
    3619:	48 8d 05 ae 2b 00 00 	lea    0x2bae(%rip),%rax        # 61ce <_ZN2nv6targetL5sm_90E+0x9e>
    3620:	48 89 c6             	mov    %rax,%rsi
    3623:	48 89 d7             	mov    %rdx,%rdi
    3626:	e8 95 ee ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    362b:	48 8b 15 9e 59 00 00 	mov    0x599e(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3632:	48 89 d6             	mov    %rdx,%rsi
    3635:	48 89 c7             	mov    %rax,%rdi
    3638:	e8 c3 ee ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    363d:	48 8d 05 94 2b 00 00 	lea    0x2b94(%rip),%rax        # 61d8 <_ZN2nv6targetL5sm_90E+0xa8>
    3644:	48 89 c6             	mov    %rax,%rsi
    3647:	48 8d 05 f2 59 00 00 	lea    0x59f2(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    364e:	48 89 c7             	mov    %rax,%rdi
    3651:	e8 6a ee ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3656:	48 8b 15 73 59 00 00 	mov    0x5973(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    365d:	48 89 d6             	mov    %rdx,%rsi
    3660:	48 89 c7             	mov    %rax,%rdi
    3663:	e8 98 ee ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3668:	48 8d 05 89 2b 00 00 	lea    0x2b89(%rip),%rax        # 61f8 <_ZN2nv6targetL5sm_90E+0xc8>
    366f:	48 89 c6             	mov    %rax,%rsi
    3672:	48 8d 05 c7 59 00 00 	lea    0x59c7(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3679:	48 89 c7             	mov    %rax,%rdi
    367c:	e8 3f ee ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3681:	48 8b 15 48 59 00 00 	mov    0x5948(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3688:	48 89 d6             	mov    %rdx,%rsi
    368b:	48 89 c7             	mov    %rax,%rdi
    368e:	e8 6d ee ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3693:	48 c7 85 60 ff ff ff 	movq   $0x0,-0xa0(%rbp)
    369a:	00 00 00 00 
    369e:	e9 2a 03 00 00       	jmp    39cd <main+0x8ae>
    36a3:	48 8b 95 60 ff ff ff 	mov    -0xa0(%rbp),%rdx
    36aa:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    36b1:	48 89 d6             	mov    %rdx,%rsi
    36b4:	48 89 c7             	mov    %rax,%rdi
    36b7:	e8 ac 0d 00 00       	call   4468 <_ZNSt6vectorI15BenchmarkResultSaIS0_EEixEm>
    36bc:	48 89 85 68 ff ff ff 	mov    %rax,-0x98(%rbp)
    36c3:	48 8d 05 30 2b 00 00 	lea    0x2b30(%rip),%rax        # 61fa <_ZN2nv6targetL5sm_90E+0xca>
    36ca:	48 89 c6             	mov    %rax,%rsi
    36cd:	48 8d 05 6c 59 00 00 	lea    0x596c(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    36d4:	48 89 c7             	mov    %rax,%rdi
    36d7:	e8 e4 ed ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    36dc:	48 8b 15 ed 58 00 00 	mov    0x58ed(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    36e3:	48 89 d6             	mov    %rdx,%rsi
    36e6:	48 89 c7             	mov    %rax,%rdi
    36e9:	e8 12 ee ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    36ee:	48 8d 05 09 2b 00 00 	lea    0x2b09(%rip),%rax        # 61fe <_ZN2nv6targetL5sm_90E+0xce>
    36f5:	48 89 c6             	mov    %rax,%rsi
    36f8:	48 8d 05 41 59 00 00 	lea    0x5941(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    36ff:	48 89 c7             	mov    %rax,%rdi
    3702:	e8 b9 ed ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3707:	48 89 c2             	mov    %rax,%rdx
    370a:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    3711:	8b 00                	mov    (%rax),%eax
    3713:	89 c6                	mov    %eax,%esi
    3715:	48 89 d7             	mov    %rdx,%rdi
    3718:	e8 03 ef ff ff       	call   2620 <_ZNSolsEi@plt>
    371d:	48 89 c2             	mov    %rax,%rdx
    3720:	48 8d 05 e1 2a 00 00 	lea    0x2ae1(%rip),%rax        # 6208 <_ZN2nv6targetL5sm_90E+0xd8>
    3727:	48 89 c6             	mov    %rax,%rsi
    372a:	48 89 d7             	mov    %rdx,%rdi
    372d:	e8 8e ed ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3732:	48 8b 15 97 58 00 00 	mov    0x5897(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3739:	48 89 d6             	mov    %rdx,%rsi
    373c:	48 89 c7             	mov    %rax,%rdi
    373f:	e8 bc ed ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3744:	48 8d 05 bf 2a 00 00 	lea    0x2abf(%rip),%rax        # 620a <_ZN2nv6targetL5sm_90E+0xda>
    374b:	48 89 c6             	mov    %rax,%rsi
    374e:	48 8d 05 eb 58 00 00 	lea    0x58eb(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3755:	48 89 c7             	mov    %rax,%rdi
    3758:	e8 63 ed ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    375d:	48 89 c2             	mov    %rax,%rdx
    3760:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    3767:	8b 40 04             	mov    0x4(%rax),%eax
    376a:	89 c6                	mov    %eax,%esi
    376c:	48 89 d7             	mov    %rdx,%rdi
    376f:	e8 ac ee ff ff       	call   2620 <_ZNSolsEi@plt>
    3774:	48 89 c2             	mov    %rax,%rdx
    3777:	48 8d 05 8a 2a 00 00 	lea    0x2a8a(%rip),%rax        # 6208 <_ZN2nv6targetL5sm_90E+0xd8>
    377e:	48 89 c6             	mov    %rax,%rsi
    3781:	48 89 d7             	mov    %rdx,%rdi
    3784:	e8 37 ed ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3789:	48 8b 15 40 58 00 00 	mov    0x5840(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3790:	48 89 d6             	mov    %rdx,%rsi
    3793:	48 89 c7             	mov    %rax,%rdi
    3796:	e8 65 ed ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    379b:	48 8d 05 72 2a 00 00 	lea    0x2a72(%rip),%rax        # 6214 <_ZN2nv6targetL5sm_90E+0xe4>
    37a2:	48 89 c6             	mov    %rax,%rsi
    37a5:	48 8d 05 94 58 00 00 	lea    0x5894(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    37ac:	48 89 c7             	mov    %rax,%rdi
    37af:	e8 0c ed ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    37b4:	48 89 c2             	mov    %rax,%rdx
    37b7:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    37be:	8b 40 08             	mov    0x8(%rax),%eax
    37c1:	89 c6                	mov    %eax,%esi
    37c3:	48 89 d7             	mov    %rdx,%rdi
    37c6:	e8 55 ee ff ff       	call   2620 <_ZNSolsEi@plt>
    37cb:	48 89 c2             	mov    %rax,%rdx
    37ce:	48 8d 05 33 2a 00 00 	lea    0x2a33(%rip),%rax        # 6208 <_ZN2nv6targetL5sm_90E+0xd8>
    37d5:	48 89 c6             	mov    %rax,%rsi
    37d8:	48 89 d7             	mov    %rdx,%rdi
    37db:	e8 e0 ec ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    37e0:	48 8b 15 e9 57 00 00 	mov    0x57e9(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    37e7:	48 89 d6             	mov    %rdx,%rsi
    37ea:	48 89 c7             	mov    %rax,%rdi
    37ed:	e8 0e ed ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    37f2:	48 8d 05 25 2a 00 00 	lea    0x2a25(%rip),%rax        # 621e <_ZN2nv6targetL5sm_90E+0xee>
    37f9:	48 89 c6             	mov    %rax,%rsi
    37fc:	48 8d 05 3d 58 00 00 	lea    0x583d(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3803:	48 89 c7             	mov    %rax,%rdi
    3806:	e8 b5 ec ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    380b:	48 89 c2             	mov    %rax,%rdx
    380e:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    3815:	8b 40 0c             	mov    0xc(%rax),%eax
    3818:	66 0f 6e c0          	movd   %eax,%xmm0
    381c:	48 89 d7             	mov    %rdx,%rdi
    381f:	e8 2c ec ff ff       	call   2450 <_ZNSolsEf@plt>
    3824:	48 89 c2             	mov    %rax,%rdx
    3827:	48 8d 05 da 29 00 00 	lea    0x29da(%rip),%rax        # 6208 <_ZN2nv6targetL5sm_90E+0xd8>
    382e:	48 89 c6             	mov    %rax,%rsi
    3831:	48 89 d7             	mov    %rdx,%rdi
    3834:	e8 87 ec ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3839:	48 8b 15 90 57 00 00 	mov    0x5790(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3840:	48 89 d6             	mov    %rdx,%rsi
    3843:	48 89 c7             	mov    %rax,%rdi
    3846:	e8 b5 ec ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    384b:	48 8d 05 df 29 00 00 	lea    0x29df(%rip),%rax        # 6231 <_ZN2nv6targetL5sm_90E+0x101>
    3852:	48 89 c6             	mov    %rax,%rsi
    3855:	48 8d 05 e4 57 00 00 	lea    0x57e4(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    385c:	48 89 c7             	mov    %rax,%rdi
    385f:	e8 5c ec ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3864:	48 89 c2             	mov    %rax,%rdx
    3867:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    386e:	48 8b 40 10          	mov    0x10(%rax),%rax
    3872:	66 48 0f 6e c0       	movq   %rax,%xmm0
    3877:	48 89 d7             	mov    %rdx,%rdi
    387a:	e8 f1 ed ff ff       	call   2670 <_ZNSolsEd@plt>
    387f:	48 89 c2             	mov    %rax,%rdx
    3882:	48 8d 05 7f 29 00 00 	lea    0x297f(%rip),%rax        # 6208 <_ZN2nv6targetL5sm_90E+0xd8>
    3889:	48 89 c6             	mov    %rax,%rsi
    388c:	48 89 d7             	mov    %rdx,%rdi
    388f:	e8 2c ec ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3894:	48 8b 15 35 57 00 00 	mov    0x5735(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    389b:	48 89 d6             	mov    %rdx,%rsi
    389e:	48 89 c7             	mov    %rax,%rdi
    38a1:	e8 5a ec ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    38a6:	48 8d 05 93 29 00 00 	lea    0x2993(%rip),%rax        # 6240 <_ZN2nv6targetL5sm_90E+0x110>
    38ad:	48 89 c6             	mov    %rax,%rsi
    38b0:	48 8d 05 89 57 00 00 	lea    0x5789(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    38b7:	48 89 c7             	mov    %rax,%rdi
    38ba:	e8 01 ec ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    38bf:	48 89 c2             	mov    %rax,%rdx
    38c2:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    38c9:	48 8b 40 18          	mov    0x18(%rax),%rax
    38cd:	66 48 0f 6e c0       	movq   %rax,%xmm0
    38d2:	48 89 d7             	mov    %rdx,%rdi
    38d5:	e8 96 ed ff ff       	call   2670 <_ZNSolsEd@plt>
    38da:	48 89 c2             	mov    %rax,%rdx
    38dd:	48 8d 05 24 29 00 00 	lea    0x2924(%rip),%rax        # 6208 <_ZN2nv6targetL5sm_90E+0xd8>
    38e4:	48 89 c6             	mov    %rax,%rsi
    38e7:	48 89 d7             	mov    %rdx,%rdi
    38ea:	e8 d1 eb ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    38ef:	48 8b 15 da 56 00 00 	mov    0x56da(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    38f6:	48 89 d6             	mov    %rdx,%rsi
    38f9:	48 89 c7             	mov    %rax,%rdi
    38fc:	e8 ff eb ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3901:	48 8d 05 49 29 00 00 	lea    0x2949(%rip),%rax        # 6251 <_ZN2nv6targetL5sm_90E+0x121>
    3908:	48 89 c6             	mov    %rax,%rsi
    390b:	48 8d 05 2e 57 00 00 	lea    0x572e(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3912:	48 89 c7             	mov    %rax,%rdi
    3915:	e8 a6 eb ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    391a:	48 89 c2             	mov    %rax,%rdx
    391d:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    3924:	0f b6 40 20          	movzbl 0x20(%rax),%eax
    3928:	84 c0                	test   %al,%al
    392a:	74 09                	je     3935 <main+0x816>
    392c:	48 8d 05 2e 29 00 00 	lea    0x292e(%rip),%rax        # 6261 <_ZN2nv6targetL5sm_90E+0x131>
    3933:	eb 07                	jmp    393c <main+0x81d>
    3935:	48 8d 05 2a 29 00 00 	lea    0x292a(%rip),%rax        # 6266 <_ZN2nv6targetL5sm_90E+0x136>
    393c:	48 89 c6             	mov    %rax,%rsi
    393f:	48 89 d7             	mov    %rdx,%rdi
    3942:	e8 79 eb ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3947:	48 8b 15 82 56 00 00 	mov    0x5682(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    394e:	48 89 d6             	mov    %rdx,%rsi
    3951:	48 89 c7             	mov    %rax,%rdi
    3954:	e8 a7 eb ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3959:	48 8d 05 0c 29 00 00 	lea    0x290c(%rip),%rax        # 626c <_ZN2nv6targetL5sm_90E+0x13c>
    3960:	48 89 c6             	mov    %rax,%rsi
    3963:	48 8d 05 d6 56 00 00 	lea    0x56d6(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    396a:	48 89 c7             	mov    %rax,%rdi
    396d:	e8 4e eb ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3972:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    3979:	48 89 c7             	mov    %rax,%rdi
    397c:	e8 b5 0a 00 00       	call   4436 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE4sizeEv>
    3981:	48 83 e8 01          	sub    $0x1,%rax
    3985:	48 39 85 60 ff ff ff 	cmp    %rax,-0xa0(%rbp)
    398c:	0f 92 c0             	setb   %al
    398f:	84 c0                	test   %al,%al
    3991:	74 19                	je     39ac <main+0x88d>
    3993:	48 8d 05 6e 28 00 00 	lea    0x286e(%rip),%rax        # 6208 <_ZN2nv6targetL5sm_90E+0xd8>
    399a:	48 89 c6             	mov    %rax,%rsi
    399d:	48 8d 05 9c 56 00 00 	lea    0x569c(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    39a4:	48 89 c7             	mov    %rax,%rdi
    39a7:	e8 14 eb ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    39ac:	48 8b 05 1d 56 00 00 	mov    0x561d(%rip),%rax        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    39b3:	48 89 c6             	mov    %rax,%rsi
    39b6:	48 8d 05 83 56 00 00 	lea    0x5683(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    39bd:	48 89 c7             	mov    %rax,%rdi
    39c0:	e8 3b eb ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    39c5:	48 83 85 60 ff ff ff 	addq   $0x1,-0xa0(%rbp)
    39cc:	01 
    39cd:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    39d4:	48 89 c7             	mov    %rax,%rdi
    39d7:	e8 5a 0a 00 00       	call   4436 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE4sizeEv>
    39dc:	48 39 85 60 ff ff ff 	cmp    %rax,-0xa0(%rbp)
    39e3:	0f 92 c0             	setb   %al
    39e6:	84 c0                	test   %al,%al
    39e8:	0f 85 b5 fc ff ff    	jne    36a3 <main+0x584>
    39ee:	48 8d 05 7b 28 00 00 	lea    0x287b(%rip),%rax        # 6270 <_ZN2nv6targetL5sm_90E+0x140>
    39f5:	48 89 c6             	mov    %rax,%rsi
    39f8:	48 8d 05 41 56 00 00 	lea    0x5641(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    39ff:	48 89 c7             	mov    %rax,%rdi
    3a02:	e8 b9 ea ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3a07:	48 8b 15 c2 55 00 00 	mov    0x55c2(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3a0e:	48 89 d6             	mov    %rdx,%rsi
    3a11:	48 89 c7             	mov    %rax,%rdi
    3a14:	e8 e7 ea ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3a19:	48 8d 05 52 28 00 00 	lea    0x2852(%rip),%rax        # 6272 <_ZN2nv6targetL5sm_90E+0x142>
    3a20:	48 89 c6             	mov    %rax,%rsi
    3a23:	48 8d 05 16 56 00 00 	lea    0x5616(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3a2a:	48 89 c7             	mov    %rax,%rdi
    3a2d:	e8 8e ea ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3a32:	48 8b 15 97 55 00 00 	mov    0x5597(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3a39:	48 89 d6             	mov    %rdx,%rsi
    3a3c:	48 89 c7             	mov    %rax,%rdi
    3a3f:	e8 bc ea ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3a44:	48 8d 05 45 28 00 00 	lea    0x2845(%rip),%rax        # 6290 <_ZN2nv6targetL5sm_90E+0x160>
    3a4b:	48 89 c6             	mov    %rax,%rsi
    3a4e:	48 8d 05 eb 55 00 00 	lea    0x55eb(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3a55:	48 89 c7             	mov    %rax,%rdi
    3a58:	e8 63 ea ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3a5d:	48 8b 15 6c 55 00 00 	mov    0x556c(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3a64:	48 89 d6             	mov    %rdx,%rsi
    3a67:	48 89 c7             	mov    %rax,%rdi
    3a6a:	e8 91 ea ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3a6f:	48 8d 05 52 28 00 00 	lea    0x2852(%rip),%rax        # 62c8 <_ZN2nv6targetL5sm_90E+0x198>
    3a76:	48 89 c6             	mov    %rax,%rsi
    3a79:	48 8d 05 c0 55 00 00 	lea    0x55c0(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3a80:	48 89 c7             	mov    %rax,%rdi
    3a83:	e8 38 ea ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3a88:	48 8b 15 41 55 00 00 	mov    0x5541(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3a8f:	48 89 d6             	mov    %rdx,%rsi
    3a92:	48 89 c7             	mov    %rax,%rdi
    3a95:	e8 66 ea ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3a9a:	48 8d 05 5f 28 00 00 	lea    0x285f(%rip),%rax        # 6300 <_ZN2nv6targetL5sm_90E+0x1d0>
    3aa1:	48 89 c6             	mov    %rax,%rsi
    3aa4:	48 8d 05 95 55 00 00 	lea    0x5595(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3aab:	48 89 c7             	mov    %rax,%rdi
    3aae:	e8 0d ea ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3ab3:	48 8b 15 16 55 00 00 	mov    0x5516(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3aba:	48 89 d6             	mov    %rdx,%rsi
    3abd:	48 89 c7             	mov    %rax,%rdi
    3ac0:	e8 3b ea ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3ac5:	48 8d 05 74 28 00 00 	lea    0x2874(%rip),%rax        # 6340 <_ZN2nv6targetL5sm_90E+0x210>
    3acc:	48 89 c6             	mov    %rax,%rsi
    3acf:	48 8d 05 6a 55 00 00 	lea    0x556a(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3ad6:	48 89 c7             	mov    %rax,%rdi
    3ad9:	e8 e2 e9 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3ade:	48 8b 15 eb 54 00 00 	mov    0x54eb(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3ae5:	48 89 d6             	mov    %rdx,%rsi
    3ae8:	48 89 c7             	mov    %rax,%rdi
    3aeb:	e8 10 ea ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3af0:	48 8d 05 81 28 00 00 	lea    0x2881(%rip),%rax        # 6378 <_ZN2nv6targetL5sm_90E+0x248>
    3af7:	48 89 c6             	mov    %rax,%rsi
    3afa:	48 8d 05 3f 55 00 00 	lea    0x553f(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3b01:	48 89 c7             	mov    %rax,%rdi
    3b04:	e8 b7 e9 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3b09:	48 8b 15 c0 54 00 00 	mov    0x54c0(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3b10:	48 89 d6             	mov    %rdx,%rsi
    3b13:	48 89 c7             	mov    %rax,%rdi
    3b16:	e8 e5 e9 ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3b1b:	48 8d 05 8e 28 00 00 	lea    0x288e(%rip),%rax        # 63b0 <_ZN2nv6targetL5sm_90E+0x280>
    3b22:	48 89 c6             	mov    %rax,%rsi
    3b25:	48 8d 05 14 55 00 00 	lea    0x5514(%rip),%rax        # 9040 <_ZSt4cout@GLIBCXX_3.4>
    3b2c:	48 89 c7             	mov    %rax,%rdi
    3b2f:	e8 8c e9 ff ff       	call   24c0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    3b34:	48 8b 15 95 54 00 00 	mov    0x5495(%rip),%rdx        # 8fd0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
    3b3b:	48 89 d6             	mov    %rdx,%rsi
    3b3e:	48 89 c7             	mov    %rax,%rdi
    3b41:	e8 ba e9 ff ff       	call   2500 <_ZNSolsEPFRSoS_E@plt>
    3b46:	bb 00 00 00 00       	mov    $0x0,%ebx
    3b4b:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    3b52:	48 89 c7             	mov    %rax,%rdi
    3b55:	e8 14 08 00 00       	call   436e <_ZNSt6vectorI15BenchmarkResultSaIS0_EED1Ev>
    3b5a:	89 d8                	mov    %ebx,%eax
    3b5c:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    3b60:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    3b67:	00 00 
    3b69:	74 5b                	je     3bc6 <main+0xaa7>
    3b6b:	eb 54                	jmp    3bc1 <main+0xaa2>
    3b6d:	f3 0f 1e fa          	endbr64
    3b71:	48 89 c3             	mov    %rax,%rbx
    3b74:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
    3b78:	48 89 c7             	mov    %rax,%rdi
    3b7b:	e8 b0 e8 ff ff       	call   2430 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@plt>
    3b80:	eb 07                	jmp    3b89 <main+0xa6a>
    3b82:	f3 0f 1e fa          	endbr64
    3b86:	48 89 c3             	mov    %rax,%rbx
    3b89:	48 8d 45 90          	lea    -0x70(%rbp),%rax
    3b8d:	48 89 c7             	mov    %rax,%rdi
    3b90:	e8 8b e9 ff ff       	call   2520 <_ZNSaIcED1Ev@plt>
    3b95:	48 89 d8             	mov    %rbx,%rax
    3b98:	48 89 c7             	mov    %rax,%rdi
    3b9b:	e8 90 ea ff ff       	call   2630 <_Unwind_Resume@plt>
    3ba0:	f3 0f 1e fa          	endbr64
    3ba4:	48 89 c3             	mov    %rax,%rbx
    3ba7:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    3bae:	48 89 c7             	mov    %rax,%rdi
    3bb1:	e8 b8 07 00 00       	call   436e <_ZNSt6vectorI15BenchmarkResultSaIS0_EED1Ev>
    3bb6:	48 89 d8             	mov    %rbx,%rax
    3bb9:	48 89 c7             	mov    %rax,%rdi
    3bbc:	e8 6f ea ff ff       	call   2630 <_Unwind_Resume@plt>
    3bc1:	e8 6a e9 ff ff       	call   2530 <__stack_chk_fail@plt>
    3bc6:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    3bca:	c9                   	leave
    3bcb:	c3                   	ret

0000000000003bcc <_ZL22____nv_dummy_param_refPv>:
    3bcc:	f3 0f 1e fa          	endbr64
    3bd0:	55                   	push   %rbp
    3bd1:	48 89 e5             	mov    %rsp,%rbp
    3bd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3bd8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3bdc:	48 89 05 8d 55 00 00 	mov    %rax,0x558d(%rip)        # 9170 <_ZZL22____nv_dummy_param_refPvE5__ref>
    3be3:	5d                   	pop    %rbp
    3be4:	c3                   	ret

0000000000003be5 <_ZL26__cudaUnregisterBinaryUtilv>:
    3be5:	f3 0f 1e fa          	endbr64
    3be9:	55                   	push   %rbp
    3bea:	48 89 e5             	mov    %rsp,%rbp
    3bed:	48 8d 05 84 55 00 00 	lea    0x5584(%rip),%rax        # 9178 <_ZL20__cudaFatCubinHandle>
    3bf4:	48 89 c7             	mov    %rax,%rdi
    3bf7:	e8 d0 ff ff ff       	call   3bcc <_ZL22____nv_dummy_param_refPv>
    3bfc:	48 8b 05 75 55 00 00 	mov    0x5575(%rip),%rax        # 9178 <_ZL20__cudaFatCubinHandle>
    3c03:	48 89 c7             	mov    %rax,%rdi
    3c06:	e8 c5 e7 ff ff       	call   23d0 <__cudaUnregisterFatBinary@plt>
    3c0b:	90                   	nop
    3c0c:	5d                   	pop    %rbp
    3c0d:	c3                   	ret

0000000000003c0e <_ZL32__nv_init_managed_rt_with_modulePPv>:
    3c0e:	f3 0f 1e fa          	endbr64
    3c12:	55                   	push   %rbp
    3c13:	48 89 e5             	mov    %rsp,%rbp
    3c16:	48 83 ec 10          	sub    $0x10,%rsp
    3c1a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3c1e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3c22:	48 89 c7             	mov    %rax,%rdi
    3c25:	e8 e6 e9 ff ff       	call   2610 <__cudaInitModule@plt>
    3c2a:	c9                   	leave
    3c2b:	c3                   	ret

0000000000003c2c <_ZL31__nv_cudaEntityRegisterCallbackPPv>:
    3c2c:	f3 0f 1e fa          	endbr64
    3c30:	55                   	push   %rbp
    3c31:	48 89 e5             	mov    %rsp,%rbp
    3c34:	48 83 ec 08          	sub    $0x8,%rsp
    3c38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3c3c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3c40:	48 89 05 39 55 00 00 	mov    %rax,0x5539(%rip)        # 9180 <_ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref>
    3c47:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3c4b:	48 89 c7             	mov    %rax,%rdi
    3c4e:	e8 16 eb ff ff       	call   2769 <_ZL37__nv_save_fatbinhandle_for_managed_rtPPv>
    3c53:	90                   	nop
    3c54:	c9                   	leave
    3c55:	c3                   	ret

0000000000003c56 <_ZL24__sti____cudaRegisterAllv>:
    3c56:	f3 0f 1e fa          	endbr64
    3c5a:	55                   	push   %rbp
    3c5b:	48 89 e5             	mov    %rsp,%rbp
    3c5e:	48 83 ec 10          	sub    $0x10,%rsp
    3c62:	48 8d 05 af 53 00 00 	lea    0x53af(%rip),%rax        # 9018 <__TMC_END__>
    3c69:	48 89 c7             	mov    %rax,%rdi
    3c6c:	e8 3f e8 ff ff       	call   24b0 <__cudaRegisterFatBinary@plt>
    3c71:	48 89 05 00 55 00 00 	mov    %rax,0x5500(%rip)        # 9178 <_ZL20__cudaFatCubinHandle>
    3c78:	48 8d 05 ad ff ff ff 	lea    -0x53(%rip),%rax        # 3c2c <_ZL31__nv_cudaEntityRegisterCallbackPPv>
    3c7f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    3c83:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    3c87:	48 8b 05 ea 54 00 00 	mov    0x54ea(%rip),%rax        # 9178 <_ZL20__cudaFatCubinHandle>
    3c8e:	48 89 c7             	mov    %rax,%rdi
    3c91:	ff d2                	call   *%rdx
    3c93:	48 8b 05 de 54 00 00 	mov    0x54de(%rip),%rax        # 9178 <_ZL20__cudaFatCubinHandle>
    3c9a:	48 89 c7             	mov    %rax,%rdi
    3c9d:	e8 ae e9 ff ff       	call   2650 <__cudaRegisterFatBinaryEnd@plt>
    3ca2:	48 8d 05 3c ff ff ff 	lea    -0xc4(%rip),%rax        # 3be5 <_ZL26__cudaUnregisterBinaryUtilv>
    3ca9:	48 89 c7             	mov    %rax,%rdi
    3cac:	e8 5f 1b 00 00       	call   5810 <atexit>
    3cb1:	90                   	nop
    3cb2:	c9                   	leave
    3cb3:	c3                   	ret

0000000000003cb4 <_Z10cudaMallocIfE9cudaErrorPPT_m>:
    3cb4:	55                   	push   %rbp
    3cb5:	48 89 e5             	mov    %rsp,%rbp
    3cb8:	48 83 ec 10          	sub    $0x10,%rsp
    3cbc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3cc0:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    3cc4:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    3cc8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3ccc:	48 89 d6             	mov    %rdx,%rsi
    3ccf:	48 89 c7             	mov    %rax,%rdi
    3cd2:	e8 a9 e6 ff ff       	call   2380 <cudaMalloc@plt>
    3cd7:	c9                   	leave
    3cd8:	c3                   	ret

0000000000003cd9 <_Z10cudaMallocI6__halfE9cudaErrorPPT_m>:
    3cd9:	55                   	push   %rbp
    3cda:	48 89 e5             	mov    %rsp,%rbp
    3cdd:	48 83 ec 10          	sub    $0x10,%rsp
    3ce1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3ce5:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    3ce9:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    3ced:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3cf1:	48 89 d6             	mov    %rdx,%rsi
    3cf4:	48 89 c7             	mov    %rax,%rdi
    3cf7:	e8 84 e6 ff ff       	call   2380 <cudaMalloc@plt>
    3cfc:	c9                   	leave
    3cfd:	c3                   	ret

0000000000003cfe <_Z41__static_initialization_and_destruction_0ii>:
    3cfe:	f3 0f 1e fa          	endbr64
    3d02:	55                   	push   %rbp
    3d03:	48 89 e5             	mov    %rsp,%rbp
    3d06:	48 83 ec 10          	sub    $0x10,%rsp
    3d0a:	89 7d fc             	mov    %edi,-0x4(%rbp)
    3d0d:	89 75 f8             	mov    %esi,-0x8(%rbp)
    3d10:	83 7d fc 01          	cmpl   $0x1,-0x4(%rbp)
    3d14:	75 3b                	jne    3d51 <_Z41__static_initialization_and_destruction_0ii+0x53>
    3d16:	81 7d f8 ff ff 00 00 	cmpl   $0xffff,-0x8(%rbp)
    3d1d:	75 32                	jne    3d51 <_Z41__static_initialization_and_destruction_0ii+0x53>
    3d1f:	48 8d 05 42 54 00 00 	lea    0x5442(%rip),%rax        # 9168 <_ZStL8__ioinit>
    3d26:	48 89 c7             	mov    %rax,%rdi
    3d29:	e8 82 e8 ff ff       	call   25b0 <_ZNSt8ios_base4InitC1Ev@plt>
    3d2e:	48 8d 05 d3 52 00 00 	lea    0x52d3(%rip),%rax        # 9008 <__dso_handle>
    3d35:	48 89 c2             	mov    %rax,%rdx
    3d38:	48 8d 05 29 54 00 00 	lea    0x5429(%rip),%rax        # 9168 <_ZStL8__ioinit>
    3d3f:	48 89 c6             	mov    %rax,%rsi
    3d42:	48 8b 05 af 52 00 00 	mov    0x52af(%rip),%rax        # 8ff8 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
    3d49:	48 89 c7             	mov    %rax,%rdi
    3d4c:	e8 3f e7 ff ff       	call   2490 <__cxa_atexit@plt>
    3d51:	90                   	nop
    3d52:	c9                   	leave
    3d53:	c3                   	ret

0000000000003d54 <_GLOBAL__sub_I__Z10elapsed_msP10CUevent_stS0_>:
    3d54:	f3 0f 1e fa          	endbr64
    3d58:	55                   	push   %rbp
    3d59:	48 89 e5             	mov    %rsp,%rbp
    3d5c:	be ff ff 00 00       	mov    $0xffff,%esi
    3d61:	bf 01 00 00 00       	mov    $0x1,%edi
    3d66:	e8 93 ff ff ff       	call   3cfe <_Z41__static_initialization_and_destruction_0ii>
    3d6b:	5d                   	pop    %rbp
    3d6c:	c3                   	ret

0000000000003d6d <_ZnwmPv>:
    3d6d:	f3 0f 1e fa          	endbr64
    3d71:	55                   	push   %rbp
    3d72:	48 89 e5             	mov    %rsp,%rbp
    3d75:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3d79:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    3d7d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    3d81:	5d                   	pop    %rbp
    3d82:	c3                   	ret
    3d83:	90                   	nop

0000000000003d84 <_ZNK6__halfcv10__half_rawEv>:
    3d84:	f3 0f 1e fa          	endbr64
    3d88:	55                   	push   %rbp
    3d89:	48 89 e5             	mov    %rsp,%rbp
    3d8c:	48 83 ec 20          	sub    $0x20,%rsp
    3d90:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    3d94:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    3d9b:	00 00 
    3d9d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    3da1:	31 c0                	xor    %eax,%eax
    3da3:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    3da7:	0f b7 00             	movzwl (%rax),%eax
    3daa:	66 89 45 f6          	mov    %ax,-0xa(%rbp)
    3dae:	0f b7 45 f6          	movzwl -0xa(%rbp),%eax
    3db2:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    3db6:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    3dbd:	00 00 
    3dbf:	74 05                	je     3dc6 <_ZNK6__halfcv10__half_rawEv+0x42>
    3dc1:	e8 6a e7 ff ff       	call   2530 <__stack_chk_fail@plt>
    3dc6:	c9                   	leave
    3dc7:	c3                   	ret

0000000000003dc8 <_ZNK6__halfcvfEv>:
    3dc8:	f3 0f 1e fa          	endbr64
    3dcc:	55                   	push   %rbp
    3dcd:	48 89 e5             	mov    %rsp,%rbp
    3dd0:	48 83 ec 10          	sub    $0x10,%rsp
    3dd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3dd8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3ddc:	0f b7 00             	movzwl (%rax),%eax
    3ddf:	89 c7                	mov    %eax,%edi
    3de1:	e8 f3 00 00 00       	call   3ed9 <_ZL12__half2float6__half>
    3de6:	c9                   	leave
    3de7:	c3                   	ret

0000000000003de8 <_ZL21__internal_half2floatt>:
    3de8:	55                   	push   %rbp
    3de9:	48 89 e5             	mov    %rsp,%rbp
    3dec:	48 83 ec 30          	sub    $0x30,%rsp
    3df0:	89 f8                	mov    %edi,%eax
    3df2:	66 89 45 dc          	mov    %ax,-0x24(%rbp)
    3df6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    3dfd:	00 00 
    3dff:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    3e03:	31 c0                	xor    %eax,%eax
    3e05:	0f b7 45 dc          	movzwl -0x24(%rbp),%eax
    3e09:	66 c1 e8 0f          	shr    $0xf,%ax
    3e0d:	0f b7 c0             	movzwl %ax,%eax
    3e10:	89 45 e8             	mov    %eax,-0x18(%rbp)
    3e13:	0f b7 45 dc          	movzwl -0x24(%rbp),%eax
    3e17:	66 c1 e8 0a          	shr    $0xa,%ax
    3e1b:	0f b7 c0             	movzwl %ax,%eax
    3e1e:	83 e0 1f             	and    $0x1f,%eax
    3e21:	89 45 ec             	mov    %eax,-0x14(%rbp)
    3e24:	0f b7 45 dc          	movzwl -0x24(%rbp),%eax
    3e28:	c1 e0 0d             	shl    $0xd,%eax
    3e2b:	25 00 e0 7f 00       	and    $0x7fe000,%eax
    3e30:	89 45 f0             	mov    %eax,-0x10(%rbp)
    3e33:	83 7d ec 1f          	cmpl   $0x1f,-0x14(%rbp)
    3e37:	75 31                	jne    3e6a <_ZL21__internal_half2floatt+0x82>
    3e39:	83 7d f0 00          	cmpl   $0x0,-0x10(%rbp)
    3e3d:	74 07                	je     3e46 <_ZL21__internal_half2floatt+0x5e>
    3e3f:	8b 45 e8             	mov    -0x18(%rbp),%eax
    3e42:	d1 e8                	shr    $1,%eax
    3e44:	eb 03                	jmp    3e49 <_ZL21__internal_half2floatt+0x61>
    3e46:	8b 45 e8             	mov    -0x18(%rbp),%eax
    3e49:	89 45 e8             	mov    %eax,-0x18(%rbp)
    3e4c:	83 7d f0 00          	cmpl   $0x0,-0x10(%rbp)
    3e50:	74 07                	je     3e59 <_ZL21__internal_half2floatt+0x71>
    3e52:	b8 ff ff 7f 00       	mov    $0x7fffff,%eax
    3e57:	eb 05                	jmp    3e5e <_ZL21__internal_half2floatt+0x76>
    3e59:	b8 00 00 00 00       	mov    $0x0,%eax
    3e5e:	89 45 f0             	mov    %eax,-0x10(%rbp)
    3e61:	c7 45 ec ff 00 00 00 	movl   $0xff,-0x14(%rbp)
    3e68:	eb 38                	jmp    3ea2 <_ZL21__internal_half2floatt+0xba>
    3e6a:	83 7d ec 00          	cmpl   $0x0,-0x14(%rbp)
    3e6e:	75 2e                	jne    3e9e <_ZL21__internal_half2floatt+0xb6>
    3e70:	83 7d f0 00          	cmpl   $0x0,-0x10(%rbp)
    3e74:	74 2c                	je     3ea2 <_ZL21__internal_half2floatt+0xba>
    3e76:	c7 45 ec 71 00 00 00 	movl   $0x71,-0x14(%rbp)
    3e7d:	8b 45 f0             	mov    -0x10(%rbp),%eax
    3e80:	25 00 00 40 00       	and    $0x400000,%eax
    3e85:	89 45 f4             	mov    %eax,-0xc(%rbp)
    3e88:	d1 65 f0             	shll   $1,-0x10(%rbp)
    3e8b:	83 6d ec 01          	subl   $0x1,-0x14(%rbp)
    3e8f:	83 7d f4 00          	cmpl   $0x0,-0xc(%rbp)
    3e93:	74 e8                	je     3e7d <_ZL21__internal_half2floatt+0x95>
    3e95:	81 65 f0 ff ff 7f 00 	andl   $0x7fffff,-0x10(%rbp)
    3e9c:	eb 04                	jmp    3ea2 <_ZL21__internal_half2floatt+0xba>
    3e9e:	83 45 ec 70          	addl   $0x70,-0x14(%rbp)
    3ea2:	8b 45 e8             	mov    -0x18(%rbp),%eax
    3ea5:	c1 e0 1f             	shl    $0x1f,%eax
    3ea8:	89 c2                	mov    %eax,%edx
    3eaa:	8b 45 ec             	mov    -0x14(%rbp),%eax
    3ead:	c1 e0 17             	shl    $0x17,%eax
    3eb0:	09 d0                	or     %edx,%eax
    3eb2:	0b 45 f0             	or     -0x10(%rbp),%eax
    3eb5:	89 45 e4             	mov    %eax,-0x1c(%rbp)
    3eb8:	8b 45 e4             	mov    -0x1c(%rbp),%eax
    3ebb:	89 45 e0             	mov    %eax,-0x20(%rbp)
    3ebe:	f3 0f 10 45 e0       	movss  -0x20(%rbp),%xmm0
    3ec3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3ec7:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    3ece:	00 00 
    3ed0:	74 05                	je     3ed7 <_ZL21__internal_half2floatt+0xef>
    3ed2:	e8 59 e6 ff ff       	call   2530 <__stack_chk_fail@plt>
    3ed7:	c9                   	leave
    3ed8:	c3                   	ret

0000000000003ed9 <_ZL12__half2float6__half>:
    3ed9:	55                   	push   %rbp
    3eda:	48 89 e5             	mov    %rsp,%rbp
    3edd:	48 83 ec 20          	sub    $0x20,%rsp
    3ee1:	66 89 7d ee          	mov    %di,-0x12(%rbp)
    3ee5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    3eec:	00 00 
    3eee:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    3ef2:	31 c0                	xor    %eax,%eax
    3ef4:	48 8d 45 ee          	lea    -0x12(%rbp),%rax
    3ef8:	48 89 c7             	mov    %rax,%rdi
    3efb:	e8 84 fe ff ff       	call   3d84 <_ZNK6__halfcv10__half_rawEv>
    3f00:	66 89 45 f2          	mov    %ax,-0xe(%rbp)
    3f04:	0f b7 45 f2          	movzwl -0xe(%rbp),%eax
    3f08:	0f b7 c0             	movzwl %ax,%eax
    3f0b:	89 c7                	mov    %eax,%edi
    3f0d:	e8 d6 fe ff ff       	call   3de8 <_ZL21__internal_half2floatt>
    3f12:	66 0f 7e c0          	movd   %xmm0,%eax
    3f16:	89 45 f4             	mov    %eax,-0xc(%rbp)
    3f19:	f3 0f 10 45 f4       	movss  -0xc(%rbp),%xmm0
    3f1e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3f22:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    3f29:	00 00 
    3f2b:	74 05                	je     3f32 <_ZL12__half2float6__half+0x59>
    3f2d:	e8 fe e5 ff ff       	call   2530 <__stack_chk_fail@plt>
    3f32:	c9                   	leave
    3f33:	c3                   	ret

0000000000003f34 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implD1Ev>:
    3f34:	f3 0f 1e fa          	endbr64
    3f38:	55                   	push   %rbp
    3f39:	48 89 e5             	mov    %rsp,%rbp
    3f3c:	48 83 ec 10          	sub    $0x10,%rsp
    3f40:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3f44:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3f48:	48 89 c7             	mov    %rax,%rdi
    3f4b:	e8 70 01 00 00       	call   40c0 <_ZNSaIfED1Ev>
    3f50:	90                   	nop
    3f51:	c9                   	leave
    3f52:	c3                   	ret
    3f53:	90                   	nop

0000000000003f54 <_ZNSt12_Vector_baseIfSaIfEEC1Ev>:
    3f54:	f3 0f 1e fa          	endbr64
    3f58:	55                   	push   %rbp
    3f59:	48 89 e5             	mov    %rsp,%rbp
    3f5c:	48 83 ec 10          	sub    $0x10,%rsp
    3f60:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3f64:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3f68:	48 89 c7             	mov    %rax,%rdi
    3f6b:	e8 24 01 00 00       	call   4094 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implC1Ev>
    3f70:	90                   	nop
    3f71:	c9                   	leave
    3f72:	c3                   	ret
    3f73:	90                   	nop

0000000000003f74 <_ZNSt6vectorIfSaIfEEC1Ev>:
    3f74:	f3 0f 1e fa          	endbr64
    3f78:	55                   	push   %rbp
    3f79:	48 89 e5             	mov    %rsp,%rbp
    3f7c:	48 83 ec 10          	sub    $0x10,%rsp
    3f80:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3f84:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3f88:	48 89 c7             	mov    %rax,%rdi
    3f8b:	e8 c4 ff ff ff       	call   3f54 <_ZNSt12_Vector_baseIfSaIfEEC1Ev>
    3f90:	90                   	nop
    3f91:	c9                   	leave
    3f92:	c3                   	ret
    3f93:	90                   	nop

0000000000003f94 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE12_Vector_implD1Ev>:
    3f94:	f3 0f 1e fa          	endbr64
    3f98:	55                   	push   %rbp
    3f99:	48 89 e5             	mov    %rsp,%rbp
    3f9c:	48 83 ec 10          	sub    $0x10,%rsp
    3fa0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3fa4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3fa8:	48 89 c7             	mov    %rax,%rdi
    3fab:	e8 42 03 00 00       	call   42f2 <_ZNSaI15BenchmarkResultED1Ev>
    3fb0:	90                   	nop
    3fb1:	c9                   	leave
    3fb2:	c3                   	ret
    3fb3:	90                   	nop

0000000000003fb4 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EEC1Ev>:
    3fb4:	f3 0f 1e fa          	endbr64
    3fb8:	55                   	push   %rbp
    3fb9:	48 89 e5             	mov    %rsp,%rbp
    3fbc:	48 83 ec 10          	sub    $0x10,%rsp
    3fc0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3fc4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3fc8:	48 89 c7             	mov    %rax,%rdi
    3fcb:	e8 f6 02 00 00       	call   42c6 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE12_Vector_implC1Ev>
    3fd0:	90                   	nop
    3fd1:	c9                   	leave
    3fd2:	c3                   	ret
    3fd3:	90                   	nop

0000000000003fd4 <_ZNSt6vectorI15BenchmarkResultSaIS0_EEC1Ev>:
    3fd4:	f3 0f 1e fa          	endbr64
    3fd8:	55                   	push   %rbp
    3fd9:	48 89 e5             	mov    %rsp,%rbp
    3fdc:	48 83 ec 10          	sub    $0x10,%rsp
    3fe0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    3fe4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    3fe8:	48 89 c7             	mov    %rax,%rdi
    3feb:	e8 c4 ff ff ff       	call   3fb4 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EEC1Ev>
    3ff0:	90                   	nop
    3ff1:	c9                   	leave
    3ff2:	c3                   	ret
    3ff3:	90                   	nop

0000000000003ff4 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev>:
    3ff4:	f3 0f 1e fa          	endbr64
    3ff8:	55                   	push   %rbp
    3ff9:	48 89 e5             	mov    %rsp,%rbp
    3ffc:	48 83 ec 10          	sub    $0x10,%rsp
    4000:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4004:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4008:	48 89 c7             	mov    %rax,%rdi
    400b:	e8 90 e3 ff ff       	call   23a0 <_ZNSaIcED2Ev@plt>
    4010:	90                   	nop
    4011:	c9                   	leave
    4012:	c3                   	ret
    4013:	90                   	nop

0000000000004014 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEmcRKS3_>:
    4014:	f3 0f 1e fa          	endbr64
    4018:	55                   	push   %rbp
    4019:	48 89 e5             	mov    %rsp,%rbp
    401c:	53                   	push   %rbx
    401d:	48 83 ec 28          	sub    $0x28,%rsp
    4021:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    4025:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    4029:	89 d0                	mov    %edx,%eax
    402b:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
    402f:	88 45 dc             	mov    %al,-0x24(%rbp)
    4032:	48 8b 5d e8          	mov    -0x18(%rbp),%rbx
    4036:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    403a:	48 89 c7             	mov    %rax,%rdi
    403d:	e8 3e e4 ff ff       	call   2480 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv@plt>
    4042:	48 89 c1             	mov    %rax,%rcx
    4045:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    4049:	48 89 c2             	mov    %rax,%rdx
    404c:	48 89 ce             	mov    %rcx,%rsi
    404f:	48 89 df             	mov    %rbx,%rdi
    4052:	e8 b9 e4 ff ff       	call   2510 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcRKS3_@plt>
    4057:	0f be 55 dc          	movsbl -0x24(%rbp),%edx
    405b:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
    405f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4063:	48 89 ce             	mov    %rcx,%rsi
    4066:	48 89 c7             	mov    %rax,%rdi
    4069:	e8 d2 e4 ff ff       	call   2540 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@plt>
    406e:	eb 1e                	jmp    408e <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEmcRKS3_+0x7a>
    4070:	f3 0f 1e fa          	endbr64
    4074:	48 89 c3             	mov    %rax,%rbx
    4077:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    407b:	48 89 c7             	mov    %rax,%rdi
    407e:	e8 71 ff ff ff       	call   3ff4 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev>
    4083:	48 89 d8             	mov    %rbx,%rax
    4086:	48 89 c7             	mov    %rax,%rdi
    4089:	e8 a2 e5 ff ff       	call   2630 <_Unwind_Resume@plt>
    408e:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    4092:	c9                   	leave
    4093:	c3                   	ret

0000000000004094 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implC1Ev>:
    4094:	f3 0f 1e fa          	endbr64
    4098:	55                   	push   %rbp
    4099:	48 89 e5             	mov    %rsp,%rbp
    409c:	48 83 ec 10          	sub    $0x10,%rsp
    40a0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    40a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    40a8:	48 89 c7             	mov    %rax,%rdi
    40ab:	e8 e6 03 00 00       	call   4496 <_ZNSaIfEC1Ev>
    40b0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    40b4:	48 89 c7             	mov    %rax,%rdi
    40b7:	e8 fa 03 00 00       	call   44b6 <_ZNSt12_Vector_baseIfSaIfEE17_Vector_impl_dataC1Ev>
    40bc:	90                   	nop
    40bd:	c9                   	leave
    40be:	c3                   	ret
    40bf:	90                   	nop

00000000000040c0 <_ZNSaIfED1Ev>:
    40c0:	f3 0f 1e fa          	endbr64
    40c4:	55                   	push   %rbp
    40c5:	48 89 e5             	mov    %rsp,%rbp
    40c8:	48 83 ec 10          	sub    $0x10,%rsp
    40cc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    40d0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    40d4:	48 89 c7             	mov    %rax,%rdi
    40d7:	e8 0c 04 00 00       	call   44e8 <_ZNSt15__new_allocatorIfED1Ev>
    40dc:	90                   	nop
    40dd:	c9                   	leave
    40de:	c3                   	ret
    40df:	90                   	nop

00000000000040e0 <_ZNSt12_Vector_baseIfSaIfEED1Ev>:
    40e0:	f3 0f 1e fa          	endbr64
    40e4:	55                   	push   %rbp
    40e5:	48 89 e5             	mov    %rsp,%rbp
    40e8:	48 83 ec 10          	sub    $0x10,%rsp
    40ec:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    40f0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    40f4:	48 8b 50 10          	mov    0x10(%rax),%rdx
    40f8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    40fc:	48 8b 00             	mov    (%rax),%rax
    40ff:	48 29 c2             	sub    %rax,%rdx
    4102:	48 89 d0             	mov    %rdx,%rax
    4105:	48 c1 f8 02          	sar    $0x2,%rax
    4109:	48 89 c2             	mov    %rax,%rdx
    410c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4110:	48 8b 08             	mov    (%rax),%rcx
    4113:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4117:	48 89 ce             	mov    %rcx,%rsi
    411a:	48 89 c7             	mov    %rax,%rdi
    411d:	e8 d6 03 00 00       	call   44f8 <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm>
    4122:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4126:	48 89 c7             	mov    %rax,%rdi
    4129:	e8 06 fe ff ff       	call   3f34 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implD1Ev>
    412e:	90                   	nop
    412f:	c9                   	leave
    4130:	c3                   	ret
    4131:	90                   	nop

0000000000004132 <_ZNSt6vectorIfSaIfEED1Ev>:
    4132:	f3 0f 1e fa          	endbr64
    4136:	55                   	push   %rbp
    4137:	48 89 e5             	mov    %rsp,%rbp
    413a:	48 83 ec 10          	sub    $0x10,%rsp
    413e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4142:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4146:	48 89 c7             	mov    %rax,%rdi
    4149:	e8 e4 03 00 00       	call   4532 <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
    414e:	48 89 c2             	mov    %rax,%rdx
    4151:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4155:	48 8b 48 08          	mov    0x8(%rax),%rcx
    4159:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    415d:	48 8b 00             	mov    (%rax),%rax
    4160:	48 89 ce             	mov    %rcx,%rsi
    4163:	48 89 c7             	mov    %rax,%rdi
    4166:	e8 d9 03 00 00       	call   4544 <_ZSt8_DestroyIPffEvT_S1_RSaIT0_E>
    416b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    416f:	48 89 c7             	mov    %rax,%rdi
    4172:	e8 69 ff ff ff       	call   40e0 <_ZNSt12_Vector_baseIfSaIfEED1Ev>
    4177:	90                   	nop
    4178:	c9                   	leave
    4179:	c3                   	ret

000000000000417a <_ZNSt6vectorIfSaIfEE9push_backEOf>:
    417a:	f3 0f 1e fa          	endbr64
    417e:	55                   	push   %rbp
    417f:	48 89 e5             	mov    %rsp,%rbp
    4182:	48 83 ec 10          	sub    $0x10,%rsp
    4186:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    418a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    418e:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    4192:	48 89 c7             	mov    %rax,%rdi
    4195:	e8 d8 03 00 00       	call   4572 <_ZSt4moveIRfEONSt16remove_referenceIT_E4typeEOS2_>
    419a:	48 89 c2             	mov    %rax,%rdx
    419d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    41a1:	48 89 d6             	mov    %rdx,%rsi
    41a4:	48 89 c7             	mov    %rax,%rdi
    41a7:	e8 d8 03 00 00       	call   4584 <_ZNSt6vectorIfSaIfEE12emplace_backIJfEEERfDpOT_>
    41ac:	90                   	nop
    41ad:	c9                   	leave
    41ae:	c3                   	ret
    41af:	90                   	nop

00000000000041b0 <_ZNSt6vectorIfSaIfEE5beginEv>:
    41b0:	f3 0f 1e fa          	endbr64
    41b4:	55                   	push   %rbp
    41b5:	48 89 e5             	mov    %rsp,%rbp
    41b8:	48 83 ec 20          	sub    $0x20,%rsp
    41bc:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    41c0:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    41c7:	00 00 
    41c9:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    41cd:	31 c0                	xor    %eax,%eax
    41cf:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    41d3:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    41d7:	48 89 d6             	mov    %rdx,%rsi
    41da:	48 89 c7             	mov    %rax,%rdi
    41dd:	e8 4a 04 00 00       	call   462c <_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC1ERKS1_>
    41e2:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    41e6:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    41ea:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    41f1:	00 00 
    41f3:	74 05                	je     41fa <_ZNSt6vectorIfSaIfEE5beginEv+0x4a>
    41f5:	e8 36 e3 ff ff       	call   2530 <__stack_chk_fail@plt>
    41fa:	c9                   	leave
    41fb:	c3                   	ret

00000000000041fc <_ZNSt6vectorIfSaIfEE3endEv>:
    41fc:	f3 0f 1e fa          	endbr64
    4200:	55                   	push   %rbp
    4201:	48 89 e5             	mov    %rsp,%rbp
    4204:	48 83 ec 20          	sub    $0x20,%rsp
    4208:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    420c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    4213:	00 00 
    4215:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    4219:	31 c0                	xor    %eax,%eax
    421b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    421f:	48 8d 50 08          	lea    0x8(%rax),%rdx
    4223:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    4227:	48 89 d6             	mov    %rdx,%rsi
    422a:	48 89 c7             	mov    %rax,%rdi
    422d:	e8 fa 03 00 00       	call   462c <_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC1ERKS1_>
    4232:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    4236:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    423a:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    4241:	00 00 
    4243:	74 05                	je     424a <_ZNSt6vectorIfSaIfEE3endEv+0x4e>
    4245:	e8 e6 e2 ff ff       	call   2530 <__stack_chk_fail@plt>
    424a:	c9                   	leave
    424b:	c3                   	ret

000000000000424c <_ZN9__gnu_cxxneIPfSt6vectorIfSaIfEEEEbRKNS_17__normal_iteratorIT_T0_EESA_>:
    424c:	f3 0f 1e fa          	endbr64
    4250:	55                   	push   %rbp
    4251:	48 89 e5             	mov    %rsp,%rbp
    4254:	53                   	push   %rbx
    4255:	48 83 ec 18          	sub    $0x18,%rsp
    4259:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    425d:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    4261:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4265:	48 89 c7             	mov    %rax,%rdi
    4268:	e8 e1 03 00 00       	call   464e <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv>
    426d:	48 8b 18             	mov    (%rax),%rbx
    4270:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    4274:	48 89 c7             	mov    %rax,%rdi
    4277:	e8 d2 03 00 00       	call   464e <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv>
    427c:	48 8b 00             	mov    (%rax),%rax
    427f:	48 39 c3             	cmp    %rax,%rbx
    4282:	0f 95 c0             	setne  %al
    4285:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    4289:	c9                   	leave
    428a:	c3                   	ret
    428b:	90                   	nop

000000000000428c <_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEppEv>:
    428c:	f3 0f 1e fa          	endbr64
    4290:	55                   	push   %rbp
    4291:	48 89 e5             	mov    %rsp,%rbp
    4294:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4298:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    429c:	48 8b 00             	mov    (%rax),%rax
    429f:	48 8d 50 04          	lea    0x4(%rax),%rdx
    42a3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    42a7:	48 89 10             	mov    %rdx,(%rax)
    42aa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    42ae:	5d                   	pop    %rbp
    42af:	c3                   	ret

00000000000042b0 <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEdeEv>:
    42b0:	f3 0f 1e fa          	endbr64
    42b4:	55                   	push   %rbp
    42b5:	48 89 e5             	mov    %rsp,%rbp
    42b8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    42bc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    42c0:	48 8b 00             	mov    (%rax),%rax
    42c3:	5d                   	pop    %rbp
    42c4:	c3                   	ret
    42c5:	90                   	nop

00000000000042c6 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE12_Vector_implC1Ev>:
    42c6:	f3 0f 1e fa          	endbr64
    42ca:	55                   	push   %rbp
    42cb:	48 89 e5             	mov    %rsp,%rbp
    42ce:	48 83 ec 10          	sub    $0x10,%rsp
    42d2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    42d6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    42da:	48 89 c7             	mov    %rax,%rdi
    42dd:	e8 7e 03 00 00       	call   4660 <_ZNSaI15BenchmarkResultEC1Ev>
    42e2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    42e6:	48 89 c7             	mov    %rax,%rdi
    42e9:	e8 92 03 00 00       	call   4680 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE17_Vector_impl_dataC1Ev>
    42ee:	90                   	nop
    42ef:	c9                   	leave
    42f0:	c3                   	ret
    42f1:	90                   	nop

00000000000042f2 <_ZNSaI15BenchmarkResultED1Ev>:
    42f2:	f3 0f 1e fa          	endbr64
    42f6:	55                   	push   %rbp
    42f7:	48 89 e5             	mov    %rsp,%rbp
    42fa:	48 83 ec 10          	sub    $0x10,%rsp
    42fe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4302:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4306:	48 89 c7             	mov    %rax,%rdi
    4309:	e8 a4 03 00 00       	call   46b2 <_ZNSt15__new_allocatorI15BenchmarkResultED1Ev>
    430e:	90                   	nop
    430f:	c9                   	leave
    4310:	c3                   	ret
    4311:	90                   	nop

0000000000004312 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EED1Ev>:
    4312:	f3 0f 1e fa          	endbr64
    4316:	55                   	push   %rbp
    4317:	48 89 e5             	mov    %rsp,%rbp
    431a:	48 83 ec 10          	sub    $0x10,%rsp
    431e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4322:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4326:	48 8b 50 10          	mov    0x10(%rax),%rdx
    432a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    432e:	48 8b 00             	mov    (%rax),%rax
    4331:	48 29 c2             	sub    %rax,%rdx
    4334:	48 c1 fa 03          	sar    $0x3,%rdx
    4338:	48 b8 cd cc cc cc cc 	movabs $0xcccccccccccccccd,%rax
    433f:	cc cc cc 
    4342:	48 0f af c2          	imul   %rdx,%rax
    4346:	48 89 c2             	mov    %rax,%rdx
    4349:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    434d:	48 8b 08             	mov    (%rax),%rcx
    4350:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4354:	48 89 ce             	mov    %rcx,%rsi
    4357:	48 89 c7             	mov    %rax,%rdi
    435a:	e8 63 03 00 00       	call   46c2 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE13_M_deallocateEPS0_m>
    435f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4363:	48 89 c7             	mov    %rax,%rdi
    4366:	e8 29 fc ff ff       	call   3f94 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE12_Vector_implD1Ev>
    436b:	90                   	nop
    436c:	c9                   	leave
    436d:	c3                   	ret

000000000000436e <_ZNSt6vectorI15BenchmarkResultSaIS0_EED1Ev>:
    436e:	f3 0f 1e fa          	endbr64
    4372:	55                   	push   %rbp
    4373:	48 89 e5             	mov    %rsp,%rbp
    4376:	48 83 ec 10          	sub    $0x10,%rsp
    437a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    437e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4382:	48 89 c7             	mov    %rax,%rdi
    4385:	e8 72 03 00 00       	call   46fc <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE19_M_get_Tp_allocatorEv>
    438a:	48 89 c2             	mov    %rax,%rdx
    438d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4391:	48 8b 48 08          	mov    0x8(%rax),%rcx
    4395:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4399:	48 8b 00             	mov    (%rax),%rax
    439c:	48 89 ce             	mov    %rcx,%rsi
    439f:	48 89 c7             	mov    %rax,%rdi
    43a2:	e8 67 03 00 00       	call   470e <_ZSt8_DestroyIP15BenchmarkResultS0_EvT_S2_RSaIT0_E>
    43a7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    43ab:	48 89 c7             	mov    %rax,%rdi
    43ae:	e8 5f ff ff ff       	call   4312 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EED1Ev>
    43b3:	90                   	nop
    43b4:	c9                   	leave
    43b5:	c3                   	ret

00000000000043b6 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE9push_backERKS0_>:
    43b6:	f3 0f 1e fa          	endbr64
    43ba:	55                   	push   %rbp
    43bb:	48 89 e5             	mov    %rsp,%rbp
    43be:	48 83 ec 10          	sub    $0x10,%rsp
    43c2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    43c6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    43ca:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    43ce:	48 8b 50 08          	mov    0x8(%rax),%rdx
    43d2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    43d6:	48 8b 40 10          	mov    0x10(%rax),%rax
    43da:	48 39 c2             	cmp    %rax,%rdx
    43dd:	74 31                	je     4410 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE9push_backERKS0_+0x5a>
    43df:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    43e3:	48 8b 48 08          	mov    0x8(%rax),%rcx
    43e7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    43eb:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    43ef:	48 89 ce             	mov    %rcx,%rsi
    43f2:	48 89 c7             	mov    %rax,%rdi
    43f5:	e8 42 03 00 00       	call   473c <_ZNSt16allocator_traitsISaI15BenchmarkResultEE9constructIS0_JRKS0_EEEvRS1_PT_DpOT0_>
    43fa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    43fe:	48 8b 40 08          	mov    0x8(%rax),%rax
    4402:	48 8d 50 28          	lea    0x28(%rax),%rdx
    4406:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    440a:	48 89 50 08          	mov    %rdx,0x8(%rax)
    440e:	eb 22                	jmp    4432 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE9push_backERKS0_+0x7c>
    4410:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4414:	48 89 c7             	mov    %rax,%rdi
    4417:	e8 5e 03 00 00       	call   477a <_ZNSt6vectorI15BenchmarkResultSaIS0_EE3endEv>
    441c:	48 89 c1             	mov    %rax,%rcx
    441f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    4423:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4427:	48 89 ce             	mov    %rcx,%rsi
    442a:	48 89 c7             	mov    %rax,%rdi
    442d:	e8 98 03 00 00       	call   47ca <_ZNSt6vectorI15BenchmarkResultSaIS0_EE17_M_realloc_insertIJRKS0_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_>
    4432:	90                   	nop
    4433:	c9                   	leave
    4434:	c3                   	ret
    4435:	90                   	nop

0000000000004436 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE4sizeEv>:
    4436:	f3 0f 1e fa          	endbr64
    443a:	55                   	push   %rbp
    443b:	48 89 e5             	mov    %rsp,%rbp
    443e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4442:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4446:	48 8b 50 08          	mov    0x8(%rax),%rdx
    444a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    444e:	48 8b 00             	mov    (%rax),%rax
    4451:	48 29 c2             	sub    %rax,%rdx
    4454:	48 c1 fa 03          	sar    $0x3,%rdx
    4458:	48 b8 cd cc cc cc cc 	movabs $0xcccccccccccccccd,%rax
    445f:	cc cc cc 
    4462:	48 0f af c2          	imul   %rdx,%rax
    4466:	5d                   	pop    %rbp
    4467:	c3                   	ret

0000000000004468 <_ZNSt6vectorI15BenchmarkResultSaIS0_EEixEm>:
    4468:	f3 0f 1e fa          	endbr64
    446c:	55                   	push   %rbp
    446d:	48 89 e5             	mov    %rsp,%rbp
    4470:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4474:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4478:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    447c:	48 8b 08             	mov    (%rax),%rcx
    447f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    4483:	48 89 d0             	mov    %rdx,%rax
    4486:	48 c1 e0 02          	shl    $0x2,%rax
    448a:	48 01 d0             	add    %rdx,%rax
    448d:	48 c1 e0 03          	shl    $0x3,%rax
    4491:	48 01 c8             	add    %rcx,%rax
    4494:	5d                   	pop    %rbp
    4495:	c3                   	ret

0000000000004496 <_ZNSaIfEC1Ev>:
    4496:	f3 0f 1e fa          	endbr64
    449a:	55                   	push   %rbp
    449b:	48 89 e5             	mov    %rsp,%rbp
    449e:	48 83 ec 10          	sub    $0x10,%rsp
    44a2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    44a6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    44aa:	48 89 c7             	mov    %rax,%rdi
    44ad:	e8 f8 04 00 00       	call   49aa <_ZNSt15__new_allocatorIfEC1Ev>
    44b2:	90                   	nop
    44b3:	c9                   	leave
    44b4:	c3                   	ret
    44b5:	90                   	nop

00000000000044b6 <_ZNSt12_Vector_baseIfSaIfEE17_Vector_impl_dataC1Ev>:
    44b6:	f3 0f 1e fa          	endbr64
    44ba:	55                   	push   %rbp
    44bb:	48 89 e5             	mov    %rsp,%rbp
    44be:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    44c2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    44c6:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
    44cd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    44d1:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
    44d8:	00 
    44d9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    44dd:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
    44e4:	00 
    44e5:	90                   	nop
    44e6:	5d                   	pop    %rbp
    44e7:	c3                   	ret

00000000000044e8 <_ZNSt15__new_allocatorIfED1Ev>:
    44e8:	f3 0f 1e fa          	endbr64
    44ec:	55                   	push   %rbp
    44ed:	48 89 e5             	mov    %rsp,%rbp
    44f0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    44f4:	90                   	nop
    44f5:	5d                   	pop    %rbp
    44f6:	c3                   	ret
    44f7:	90                   	nop

00000000000044f8 <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm>:
    44f8:	f3 0f 1e fa          	endbr64
    44fc:	55                   	push   %rbp
    44fd:	48 89 e5             	mov    %rsp,%rbp
    4500:	48 83 ec 20          	sub    $0x20,%rsp
    4504:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4508:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    450c:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    4510:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    4515:	74 17                	je     452e <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm+0x36>
    4517:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    451b:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    451f:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    4523:	48 89 ce             	mov    %rcx,%rsi
    4526:	48 89 c7             	mov    %rax,%rdi
    4529:	e8 8b 04 00 00       	call   49b9 <_ZNSt16allocator_traitsISaIfEE10deallocateERS0_Pfm>
    452e:	90                   	nop
    452f:	c9                   	leave
    4530:	c3                   	ret
    4531:	90                   	nop

0000000000004532 <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>:
    4532:	f3 0f 1e fa          	endbr64
    4536:	55                   	push   %rbp
    4537:	48 89 e5             	mov    %rsp,%rbp
    453a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    453e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4542:	5d                   	pop    %rbp
    4543:	c3                   	ret

0000000000004544 <_ZSt8_DestroyIPffEvT_S1_RSaIT0_E>:
    4544:	f3 0f 1e fa          	endbr64
    4548:	55                   	push   %rbp
    4549:	48 89 e5             	mov    %rsp,%rbp
    454c:	48 83 ec 20          	sub    $0x20,%rsp
    4550:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4554:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4558:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    455c:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    4560:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4564:	48 89 d6             	mov    %rdx,%rsi
    4567:	48 89 c7             	mov    %rax,%rdi
    456a:	e8 7c 04 00 00       	call   49eb <_ZSt8_DestroyIPfEvT_S1_>
    456f:	90                   	nop
    4570:	c9                   	leave
    4571:	c3                   	ret

0000000000004572 <_ZSt4moveIRfEONSt16remove_referenceIT_E4typeEOS2_>:
    4572:	f3 0f 1e fa          	endbr64
    4576:	55                   	push   %rbp
    4577:	48 89 e5             	mov    %rsp,%rbp
    457a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    457e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4582:	5d                   	pop    %rbp
    4583:	c3                   	ret

0000000000004584 <_ZNSt6vectorIfSaIfEE12emplace_backIJfEEERfDpOT_>:
    4584:	f3 0f 1e fa          	endbr64
    4588:	55                   	push   %rbp
    4589:	48 89 e5             	mov    %rsp,%rbp
    458c:	53                   	push   %rbx
    458d:	48 83 ec 18          	sub    $0x18,%rsp
    4591:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    4595:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    4599:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    459d:	48 8b 50 08          	mov    0x8(%rax),%rdx
    45a1:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    45a5:	48 8b 40 10          	mov    0x10(%rax),%rax
    45a9:	48 39 c2             	cmp    %rax,%rdx
    45ac:	74 3c                	je     45ea <_ZNSt6vectorIfSaIfEE12emplace_backIJfEEERfDpOT_+0x66>
    45ae:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    45b2:	48 89 c7             	mov    %rax,%rdi
    45b5:	e8 5b 04 00 00       	call   4a15 <_ZSt7forwardIfEOT_RNSt16remove_referenceIS0_E4typeE>
    45ba:	48 89 c2             	mov    %rax,%rdx
    45bd:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    45c1:	48 8b 48 08          	mov    0x8(%rax),%rcx
    45c5:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    45c9:	48 89 ce             	mov    %rcx,%rsi
    45cc:	48 89 c7             	mov    %rax,%rdi
    45cf:	e8 53 04 00 00       	call   4a27 <_ZNSt16allocator_traitsISaIfEE9constructIfJfEEEvRS0_PT_DpOT0_>
    45d4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    45d8:	48 8b 40 08          	mov    0x8(%rax),%rax
    45dc:	48 8d 50 04          	lea    0x4(%rax),%rdx
    45e0:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    45e4:	48 89 50 08          	mov    %rdx,0x8(%rax)
    45e8:	eb 30                	jmp    461a <_ZNSt6vectorIfSaIfEE12emplace_backIJfEEERfDpOT_+0x96>
    45ea:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    45ee:	48 89 c7             	mov    %rax,%rdi
    45f1:	e8 1f 04 00 00       	call   4a15 <_ZSt7forwardIfEOT_RNSt16remove_referenceIS0_E4typeE>
    45f6:	48 89 c3             	mov    %rax,%rbx
    45f9:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    45fd:	48 89 c7             	mov    %rax,%rdi
    4600:	e8 f7 fb ff ff       	call   41fc <_ZNSt6vectorIfSaIfEE3endEv>
    4605:	48 89 c1             	mov    %rax,%rcx
    4608:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    460c:	48 89 da             	mov    %rbx,%rdx
    460f:	48 89 ce             	mov    %rcx,%rsi
    4612:	48 89 c7             	mov    %rax,%rdi
    4615:	e8 4a 04 00 00       	call   4a64 <_ZNSt6vectorIfSaIfEE17_M_realloc_insertIJfEEEvN9__gnu_cxx17__normal_iteratorIPfS1_EEDpOT_>
    461a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    461e:	48 89 c7             	mov    %rax,%rdi
    4621:	e8 f8 05 00 00       	call   4c1e <_ZNSt6vectorIfSaIfEE4backEv>
    4626:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    462a:	c9                   	leave
    462b:	c3                   	ret

000000000000462c <_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC1ERKS1_>:
    462c:	f3 0f 1e fa          	endbr64
    4630:	55                   	push   %rbp
    4631:	48 89 e5             	mov    %rsp,%rbp
    4634:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4638:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    463c:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    4640:	48 8b 10             	mov    (%rax),%rdx
    4643:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4647:	48 89 10             	mov    %rdx,(%rax)
    464a:	90                   	nop
    464b:	5d                   	pop    %rbp
    464c:	c3                   	ret
    464d:	90                   	nop

000000000000464e <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv>:
    464e:	f3 0f 1e fa          	endbr64
    4652:	55                   	push   %rbp
    4653:	48 89 e5             	mov    %rsp,%rbp
    4656:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    465a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    465e:	5d                   	pop    %rbp
    465f:	c3                   	ret

0000000000004660 <_ZNSaI15BenchmarkResultEC1Ev>:
    4660:	f3 0f 1e fa          	endbr64
    4664:	55                   	push   %rbp
    4665:	48 89 e5             	mov    %rsp,%rbp
    4668:	48 83 ec 10          	sub    $0x10,%rsp
    466c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4670:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4674:	48 89 c7             	mov    %rax,%rdi
    4677:	e8 08 06 00 00       	call   4c84 <_ZNSt15__new_allocatorI15BenchmarkResultEC1Ev>
    467c:	90                   	nop
    467d:	c9                   	leave
    467e:	c3                   	ret
    467f:	90                   	nop

0000000000004680 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE17_Vector_impl_dataC1Ev>:
    4680:	f3 0f 1e fa          	endbr64
    4684:	55                   	push   %rbp
    4685:	48 89 e5             	mov    %rsp,%rbp
    4688:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    468c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4690:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
    4697:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    469b:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
    46a2:	00 
    46a3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    46a7:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
    46ae:	00 
    46af:	90                   	nop
    46b0:	5d                   	pop    %rbp
    46b1:	c3                   	ret

00000000000046b2 <_ZNSt15__new_allocatorI15BenchmarkResultED1Ev>:
    46b2:	f3 0f 1e fa          	endbr64
    46b6:	55                   	push   %rbp
    46b7:	48 89 e5             	mov    %rsp,%rbp
    46ba:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    46be:	90                   	nop
    46bf:	5d                   	pop    %rbp
    46c0:	c3                   	ret
    46c1:	90                   	nop

00000000000046c2 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE13_M_deallocateEPS0_m>:
    46c2:	f3 0f 1e fa          	endbr64
    46c6:	55                   	push   %rbp
    46c7:	48 89 e5             	mov    %rsp,%rbp
    46ca:	48 83 ec 20          	sub    $0x20,%rsp
    46ce:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    46d2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    46d6:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    46da:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    46df:	74 17                	je     46f8 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE13_M_deallocateEPS0_m+0x36>
    46e1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    46e5:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    46e9:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    46ed:	48 89 ce             	mov    %rcx,%rsi
    46f0:	48 89 c7             	mov    %rax,%rdi
    46f3:	e8 9b 05 00 00       	call   4c93 <_ZNSt16allocator_traitsISaI15BenchmarkResultEE10deallocateERS1_PS0_m>
    46f8:	90                   	nop
    46f9:	c9                   	leave
    46fa:	c3                   	ret
    46fb:	90                   	nop

00000000000046fc <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE19_M_get_Tp_allocatorEv>:
    46fc:	f3 0f 1e fa          	endbr64
    4700:	55                   	push   %rbp
    4701:	48 89 e5             	mov    %rsp,%rbp
    4704:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4708:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    470c:	5d                   	pop    %rbp
    470d:	c3                   	ret

000000000000470e <_ZSt8_DestroyIP15BenchmarkResultS0_EvT_S2_RSaIT0_E>:
    470e:	f3 0f 1e fa          	endbr64
    4712:	55                   	push   %rbp
    4713:	48 89 e5             	mov    %rsp,%rbp
    4716:	48 83 ec 20          	sub    $0x20,%rsp
    471a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    471e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4722:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    4726:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    472a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    472e:	48 89 d6             	mov    %rdx,%rsi
    4731:	48 89 c7             	mov    %rax,%rdi
    4734:	e8 8c 05 00 00       	call   4cc5 <_ZSt8_DestroyIP15BenchmarkResultEvT_S2_>
    4739:	90                   	nop
    473a:	c9                   	leave
    473b:	c3                   	ret

000000000000473c <_ZNSt16allocator_traitsISaI15BenchmarkResultEE9constructIS0_JRKS0_EEEvRS1_PT_DpOT0_>:
    473c:	f3 0f 1e fa          	endbr64
    4740:	55                   	push   %rbp
    4741:	48 89 e5             	mov    %rsp,%rbp
    4744:	48 83 ec 20          	sub    $0x20,%rsp
    4748:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    474c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4750:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    4754:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4758:	48 89 c7             	mov    %rax,%rdi
    475b:	e8 8f 05 00 00       	call   4cef <_ZSt7forwardIRK15BenchmarkResultEOT_RNSt16remove_referenceIS3_E4typeE>
    4760:	48 89 c2             	mov    %rax,%rdx
    4763:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    4767:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    476b:	48 89 ce             	mov    %rcx,%rsi
    476e:	48 89 c7             	mov    %rax,%rdi
    4771:	e8 8c 05 00 00       	call   4d02 <_ZNSt15__new_allocatorI15BenchmarkResultE9constructIS0_JRKS0_EEEvPT_DpOT0_>
    4776:	90                   	nop
    4777:	c9                   	leave
    4778:	c3                   	ret
    4779:	90                   	nop

000000000000477a <_ZNSt6vectorI15BenchmarkResultSaIS0_EE3endEv>:
    477a:	f3 0f 1e fa          	endbr64
    477e:	55                   	push   %rbp
    477f:	48 89 e5             	mov    %rsp,%rbp
    4782:	48 83 ec 20          	sub    $0x20,%rsp
    4786:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    478a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    4791:	00 00 
    4793:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    4797:	31 c0                	xor    %eax,%eax
    4799:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    479d:	48 8d 50 08          	lea    0x8(%rax),%rdx
    47a1:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    47a5:	48 89 d6             	mov    %rdx,%rsi
    47a8:	48 89 c7             	mov    %rax,%rdi
    47ab:	e8 b8 05 00 00       	call   4d68 <_ZN9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEEC1ERKS2_>
    47b0:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    47b4:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    47b8:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    47bf:	00 00 
    47c1:	74 05                	je     47c8 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE3endEv+0x4e>
    47c3:	e8 68 dd ff ff       	call   2530 <__stack_chk_fail@plt>
    47c8:	c9                   	leave
    47c9:	c3                   	ret

00000000000047ca <_ZNSt6vectorI15BenchmarkResultSaIS0_EE17_M_realloc_insertIJRKS0_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_>:
    47ca:	f3 0f 1e fa          	endbr64
    47ce:	55                   	push   %rbp
    47cf:	48 89 e5             	mov    %rsp,%rbp
    47d2:	53                   	push   %rbx
    47d3:	48 83 ec 68          	sub    $0x68,%rsp
    47d7:	48 89 7d a8          	mov    %rdi,-0x58(%rbp)
    47db:	48 89 75 a0          	mov    %rsi,-0x60(%rbp)
    47df:	48 89 55 98          	mov    %rdx,-0x68(%rbp)
    47e3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    47ea:	00 00 
    47ec:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    47f0:	31 c0                	xor    %eax,%eax
    47f2:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    47f6:	48 8d 15 f0 1b 00 00 	lea    0x1bf0(%rip),%rdx        # 63ed <_ZN2nv6targetL5sm_90E+0x2bd>
    47fd:	be 01 00 00 00       	mov    $0x1,%esi
    4802:	48 89 c7             	mov    %rax,%rdi
    4805:	e8 80 05 00 00       	call   4d8a <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE12_M_check_lenEmPKc>
    480a:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
    480e:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4812:	48 8b 00             	mov    (%rax),%rax
    4815:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
    4819:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    481d:	48 8b 40 08          	mov    0x8(%rax),%rax
    4821:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    4825:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4829:	48 89 c7             	mov    %rax,%rdi
    482c:	e8 49 06 00 00       	call   4e7a <_ZNSt6vectorI15BenchmarkResultSaIS0_EE5beginEv>
    4831:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
    4835:	48 8d 55 b0          	lea    -0x50(%rbp),%rdx
    4839:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
    483d:	48 89 d6             	mov    %rdx,%rsi
    4840:	48 89 c7             	mov    %rax,%rdi
    4843:	e8 7e 06 00 00       	call   4ec6 <_ZN9__gnu_cxxmiIP15BenchmarkResultSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_>
    4848:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
    484c:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4850:	48 8b 55 b8          	mov    -0x48(%rbp),%rdx
    4854:	48 89 d6             	mov    %rdx,%rsi
    4857:	48 89 c7             	mov    %rax,%rdi
    485a:	e8 b9 06 00 00       	call   4f18 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE11_M_allocateEm>
    485f:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    4863:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4867:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    486b:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    486f:	48 89 c7             	mov    %rax,%rdi
    4872:	e8 78 04 00 00       	call   4cef <_ZSt7forwardIRK15BenchmarkResultEOT_RNSt16remove_referenceIS3_E4typeE>
    4877:	48 89 c6             	mov    %rax,%rsi
    487a:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
    487e:	48 89 d0             	mov    %rdx,%rax
    4881:	48 c1 e0 02          	shl    $0x2,%rax
    4885:	48 01 d0             	add    %rdx,%rax
    4888:	48 c1 e0 03          	shl    $0x3,%rax
    488c:	48 89 c2             	mov    %rax,%rdx
    488f:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4893:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
    4897:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    489b:	48 89 f2             	mov    %rsi,%rdx
    489e:	48 89 ce             	mov    %rcx,%rsi
    48a1:	48 89 c7             	mov    %rax,%rdi
    48a4:	e8 93 fe ff ff       	call   473c <_ZNSt16allocator_traitsISaI15BenchmarkResultEE9constructIS0_JRKS0_EEEvRS1_PT_DpOT0_>
    48a9:	48 c7 45 e0 00 00 00 	movq   $0x0,-0x20(%rbp)
    48b0:	00 
    48b1:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    48b5:	48 89 c7             	mov    %rax,%rdi
    48b8:	e8 3f fe ff ff       	call   46fc <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE19_M_get_Tp_allocatorEv>
    48bd:	48 89 c3             	mov    %rax,%rbx
    48c0:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
    48c4:	48 89 c7             	mov    %rax,%rdi
    48c7:	e8 ba 06 00 00       	call   4f86 <_ZNK9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEE4baseEv>
    48cc:	48 8b 30             	mov    (%rax),%rsi
    48cf:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
    48d3:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
    48d7:	48 89 d9             	mov    %rbx,%rcx
    48da:	48 89 c7             	mov    %rax,%rdi
    48dd:	e8 6d 06 00 00       	call   4f4f <_ZNSt6vectorI15BenchmarkResultSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_>
    48e2:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    48e6:	48 83 45 e0 28       	addq   $0x28,-0x20(%rbp)
    48eb:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    48ef:	48 89 c7             	mov    %rax,%rdi
    48f2:	e8 05 fe ff ff       	call   46fc <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE19_M_get_Tp_allocatorEv>
    48f7:	48 89 c3             	mov    %rax,%rbx
    48fa:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
    48fe:	48 89 c7             	mov    %rax,%rdi
    4901:	e8 80 06 00 00       	call   4f86 <_ZNK9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEE4baseEv>
    4906:	48 8b 00             	mov    (%rax),%rax
    4909:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    490d:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
    4911:	48 89 d9             	mov    %rbx,%rcx
    4914:	48 89 c7             	mov    %rax,%rdi
    4917:	e8 33 06 00 00       	call   4f4f <_ZNSt6vectorI15BenchmarkResultSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_>
    491c:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    4920:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4924:	48 8b 55 a8          	mov    -0x58(%rbp),%rdx
    4928:	48 8b 52 10          	mov    0x10(%rdx),%rdx
    492c:	48 2b 55 c0          	sub    -0x40(%rbp),%rdx
    4930:	48 89 d1             	mov    %rdx,%rcx
    4933:	48 c1 f9 03          	sar    $0x3,%rcx
    4937:	48 ba cd cc cc cc cc 	movabs $0xcccccccccccccccd,%rdx
    493e:	cc cc cc 
    4941:	48 0f af d1          	imul   %rcx,%rdx
    4945:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
    4949:	48 89 ce             	mov    %rcx,%rsi
    494c:	48 89 c7             	mov    %rax,%rdi
    494f:	e8 6e fd ff ff       	call   46c2 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE13_M_deallocateEPS0_m>
    4954:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4958:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
    495c:	48 89 10             	mov    %rdx,(%rax)
    495f:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4963:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    4967:	48 89 50 08          	mov    %rdx,0x8(%rax)
    496b:	48 8b 55 b8          	mov    -0x48(%rbp),%rdx
    496f:	48 89 d0             	mov    %rdx,%rax
    4972:	48 c1 e0 02          	shl    $0x2,%rax
    4976:	48 01 d0             	add    %rdx,%rax
    4979:	48 c1 e0 03          	shl    $0x3,%rax
    497d:	48 89 c2             	mov    %rax,%rdx
    4980:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4984:	48 01 c2             	add    %rax,%rdx
    4987:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    498b:	48 89 50 10          	mov    %rdx,0x10(%rax)
    498f:	90                   	nop
    4990:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4994:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    499b:	00 00 
    499d:	74 05                	je     49a4 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE17_M_realloc_insertIJRKS0_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_+0x1da>
    499f:	e8 8c db ff ff       	call   2530 <__stack_chk_fail@plt>
    49a4:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    49a8:	c9                   	leave
    49a9:	c3                   	ret

00000000000049aa <_ZNSt15__new_allocatorIfEC1Ev>:
    49aa:	f3 0f 1e fa          	endbr64
    49ae:	55                   	push   %rbp
    49af:	48 89 e5             	mov    %rsp,%rbp
    49b2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    49b6:	90                   	nop
    49b7:	5d                   	pop    %rbp
    49b8:	c3                   	ret

00000000000049b9 <_ZNSt16allocator_traitsISaIfEE10deallocateERS0_Pfm>:
    49b9:	f3 0f 1e fa          	endbr64
    49bd:	55                   	push   %rbp
    49be:	48 89 e5             	mov    %rsp,%rbp
    49c1:	48 83 ec 20          	sub    $0x20,%rsp
    49c5:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    49c9:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    49cd:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    49d1:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    49d5:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    49d9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    49dd:	48 89 ce             	mov    %rcx,%rsi
    49e0:	48 89 c7             	mov    %rax,%rdi
    49e3:	e8 b0 05 00 00       	call   4f98 <_ZNSt15__new_allocatorIfE10deallocateEPfm>
    49e8:	90                   	nop
    49e9:	c9                   	leave
    49ea:	c3                   	ret

00000000000049eb <_ZSt8_DestroyIPfEvT_S1_>:
    49eb:	f3 0f 1e fa          	endbr64
    49ef:	55                   	push   %rbp
    49f0:	48 89 e5             	mov    %rsp,%rbp
    49f3:	48 83 ec 10          	sub    $0x10,%rsp
    49f7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    49fb:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    49ff:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    4a03:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4a07:	48 89 d6             	mov    %rdx,%rsi
    4a0a:	48 89 c7             	mov    %rax,%rdi
    4a0d:	e8 bb 05 00 00       	call   4fcd <_ZNSt12_Destroy_auxILb1EE9__destroyIPfEEvT_S3_>
    4a12:	90                   	nop
    4a13:	c9                   	leave
    4a14:	c3                   	ret

0000000000004a15 <_ZSt7forwardIfEOT_RNSt16remove_referenceIS0_E4typeE>:
    4a15:	f3 0f 1e fa          	endbr64
    4a19:	55                   	push   %rbp
    4a1a:	48 89 e5             	mov    %rsp,%rbp
    4a1d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4a21:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4a25:	5d                   	pop    %rbp
    4a26:	c3                   	ret

0000000000004a27 <_ZNSt16allocator_traitsISaIfEE9constructIfJfEEEvRS0_PT_DpOT0_>:
    4a27:	f3 0f 1e fa          	endbr64
    4a2b:	55                   	push   %rbp
    4a2c:	48 89 e5             	mov    %rsp,%rbp
    4a2f:	48 83 ec 20          	sub    $0x20,%rsp
    4a33:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4a37:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4a3b:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    4a3f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4a43:	48 89 c7             	mov    %rax,%rdi
    4a46:	e8 ca ff ff ff       	call   4a15 <_ZSt7forwardIfEOT_RNSt16remove_referenceIS0_E4typeE>
    4a4b:	48 89 c2             	mov    %rax,%rdx
    4a4e:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    4a52:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4a56:	48 89 ce             	mov    %rcx,%rsi
    4a59:	48 89 c7             	mov    %rax,%rdi
    4a5c:	e8 7f 05 00 00       	call   4fe0 <_ZNSt15__new_allocatorIfE9constructIfJfEEEvPT_DpOT0_>
    4a61:	90                   	nop
    4a62:	c9                   	leave
    4a63:	c3                   	ret

0000000000004a64 <_ZNSt6vectorIfSaIfEE17_M_realloc_insertIJfEEEvN9__gnu_cxx17__normal_iteratorIPfS1_EEDpOT_>:
    4a64:	f3 0f 1e fa          	endbr64
    4a68:	55                   	push   %rbp
    4a69:	48 89 e5             	mov    %rsp,%rbp
    4a6c:	53                   	push   %rbx
    4a6d:	48 83 ec 68          	sub    $0x68,%rsp
    4a71:	48 89 7d a8          	mov    %rdi,-0x58(%rbp)
    4a75:	48 89 75 a0          	mov    %rsi,-0x60(%rbp)
    4a79:	48 89 55 98          	mov    %rdx,-0x68(%rbp)
    4a7d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    4a84:	00 00 
    4a86:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    4a8a:	31 c0                	xor    %eax,%eax
    4a8c:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4a90:	48 8d 15 56 19 00 00 	lea    0x1956(%rip),%rdx        # 63ed <_ZN2nv6targetL5sm_90E+0x2bd>
    4a97:	be 01 00 00 00       	mov    $0x1,%esi
    4a9c:	48 89 c7             	mov    %rax,%rdi
    4a9f:	e8 84 05 00 00       	call   5028 <_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc>
    4aa4:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
    4aa8:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4aac:	48 8b 00             	mov    (%rax),%rax
    4aaf:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
    4ab3:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4ab7:	48 8b 40 08          	mov    0x8(%rax),%rax
    4abb:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    4abf:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4ac3:	48 89 c7             	mov    %rax,%rdi
    4ac6:	e8 e5 f6 ff ff       	call   41b0 <_ZNSt6vectorIfSaIfEE5beginEv>
    4acb:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
    4acf:	48 8d 55 b0          	lea    -0x50(%rbp),%rdx
    4ad3:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
    4ad7:	48 89 d6             	mov    %rdx,%rsi
    4ada:	48 89 c7             	mov    %rax,%rdi
    4add:	e8 35 06 00 00       	call   5117 <_ZN9__gnu_cxxmiIPfSt6vectorIfSaIfEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_>
    4ae2:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
    4ae6:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4aea:	48 8b 55 b8          	mov    -0x48(%rbp),%rdx
    4aee:	48 89 d6             	mov    %rdx,%rsi
    4af1:	48 89 c7             	mov    %rax,%rdi
    4af4:	e8 65 06 00 00       	call   515e <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm>
    4af9:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    4afd:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4b01:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    4b05:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    4b09:	48 89 c7             	mov    %rax,%rdi
    4b0c:	e8 04 ff ff ff       	call   4a15 <_ZSt7forwardIfEOT_RNSt16remove_referenceIS0_E4typeE>
    4b11:	48 89 c2             	mov    %rax,%rdx
    4b14:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    4b18:	48 8d 0c 85 00 00 00 	lea    0x0(,%rax,4),%rcx
    4b1f:	00 
    4b20:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4b24:	48 01 c1             	add    %rax,%rcx
    4b27:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4b2b:	48 89 ce             	mov    %rcx,%rsi
    4b2e:	48 89 c7             	mov    %rax,%rdi
    4b31:	e8 f1 fe ff ff       	call   4a27 <_ZNSt16allocator_traitsISaIfEE9constructIfJfEEEvRS0_PT_DpOT0_>
    4b36:	48 c7 45 e0 00 00 00 	movq   $0x0,-0x20(%rbp)
    4b3d:	00 
    4b3e:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4b42:	48 89 c7             	mov    %rax,%rdi
    4b45:	e8 e8 f9 ff ff       	call   4532 <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
    4b4a:	48 89 c3             	mov    %rax,%rbx
    4b4d:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
    4b51:	48 89 c7             	mov    %rax,%rdi
    4b54:	e8 f5 fa ff ff       	call   464e <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv>
    4b59:	48 8b 30             	mov    (%rax),%rsi
    4b5c:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
    4b60:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
    4b64:	48 89 d9             	mov    %rbx,%rcx
    4b67:	48 89 c7             	mov    %rax,%rdi
    4b6a:	e8 26 06 00 00       	call   5195 <_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_>
    4b6f:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    4b73:	48 83 45 e0 04       	addq   $0x4,-0x20(%rbp)
    4b78:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4b7c:	48 89 c7             	mov    %rax,%rdi
    4b7f:	e8 ae f9 ff ff       	call   4532 <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
    4b84:	48 89 c3             	mov    %rax,%rbx
    4b87:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
    4b8b:	48 89 c7             	mov    %rax,%rdi
    4b8e:	e8 bb fa ff ff       	call   464e <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv>
    4b93:	48 8b 00             	mov    (%rax),%rax
    4b96:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    4b9a:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
    4b9e:	48 89 d9             	mov    %rbx,%rcx
    4ba1:	48 89 c7             	mov    %rax,%rdi
    4ba4:	e8 ec 05 00 00       	call   5195 <_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_>
    4ba9:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    4bad:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4bb1:	48 8b 55 a8          	mov    -0x58(%rbp),%rdx
    4bb5:	48 8b 52 10          	mov    0x10(%rdx),%rdx
    4bb9:	48 2b 55 c0          	sub    -0x40(%rbp),%rdx
    4bbd:	48 c1 fa 02          	sar    $0x2,%rdx
    4bc1:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
    4bc5:	48 89 ce             	mov    %rcx,%rsi
    4bc8:	48 89 c7             	mov    %rax,%rdi
    4bcb:	e8 28 f9 ff ff       	call   44f8 <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm>
    4bd0:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4bd4:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
    4bd8:	48 89 10             	mov    %rdx,(%rax)
    4bdb:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4bdf:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    4be3:	48 89 50 08          	mov    %rdx,0x8(%rax)
    4be7:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
    4beb:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    4bf2:	00 
    4bf3:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4bf7:	48 01 c2             	add    %rax,%rdx
    4bfa:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    4bfe:	48 89 50 10          	mov    %rdx,0x10(%rax)
    4c02:	90                   	nop
    4c03:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4c07:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    4c0e:	00 00 
    4c10:	74 05                	je     4c17 <_ZNSt6vectorIfSaIfEE17_M_realloc_insertIJfEEEvN9__gnu_cxx17__normal_iteratorIPfS1_EEDpOT_+0x1b3>
    4c12:	e8 19 d9 ff ff       	call   2530 <__stack_chk_fail@plt>
    4c17:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    4c1b:	c9                   	leave
    4c1c:	c3                   	ret
    4c1d:	90                   	nop

0000000000004c1e <_ZNSt6vectorIfSaIfEE4backEv>:
    4c1e:	f3 0f 1e fa          	endbr64
    4c22:	55                   	push   %rbp
    4c23:	48 89 e5             	mov    %rsp,%rbp
    4c26:	48 83 ec 30          	sub    $0x30,%rsp
    4c2a:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    4c2e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    4c35:	00 00 
    4c37:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    4c3b:	31 c0                	xor    %eax,%eax
    4c3d:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4c41:	48 89 c7             	mov    %rax,%rdi
    4c44:	e8 b3 f5 ff ff       	call   41fc <_ZNSt6vectorIfSaIfEE3endEv>
    4c49:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    4c4d:	48 8d 45 e8          	lea    -0x18(%rbp),%rax
    4c51:	be 01 00 00 00       	mov    $0x1,%esi
    4c56:	48 89 c7             	mov    %rax,%rdi
    4c59:	e8 6e 05 00 00       	call   51cc <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEmiEl>
    4c5e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    4c62:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    4c66:	48 89 c7             	mov    %rax,%rdi
    4c69:	e8 42 f6 ff ff       	call   42b0 <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEdeEv>
    4c6e:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    4c72:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    4c79:	00 00 
    4c7b:	74 05                	je     4c82 <_ZNSt6vectorIfSaIfEE4backEv+0x64>
    4c7d:	e8 ae d8 ff ff       	call   2530 <__stack_chk_fail@plt>
    4c82:	c9                   	leave
    4c83:	c3                   	ret

0000000000004c84 <_ZNSt15__new_allocatorI15BenchmarkResultEC1Ev>:
    4c84:	f3 0f 1e fa          	endbr64
    4c88:	55                   	push   %rbp
    4c89:	48 89 e5             	mov    %rsp,%rbp
    4c8c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4c90:	90                   	nop
    4c91:	5d                   	pop    %rbp
    4c92:	c3                   	ret

0000000000004c93 <_ZNSt16allocator_traitsISaI15BenchmarkResultEE10deallocateERS1_PS0_m>:
    4c93:	f3 0f 1e fa          	endbr64
    4c97:	55                   	push   %rbp
    4c98:	48 89 e5             	mov    %rsp,%rbp
    4c9b:	48 83 ec 20          	sub    $0x20,%rsp
    4c9f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4ca3:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4ca7:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    4cab:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    4caf:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    4cb3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4cb7:	48 89 ce             	mov    %rcx,%rsi
    4cba:	48 89 c7             	mov    %rax,%rdi
    4cbd:	e8 74 05 00 00       	call   5236 <_ZNSt15__new_allocatorI15BenchmarkResultE10deallocateEPS0_m>
    4cc2:	90                   	nop
    4cc3:	c9                   	leave
    4cc4:	c3                   	ret

0000000000004cc5 <_ZSt8_DestroyIP15BenchmarkResultEvT_S2_>:
    4cc5:	f3 0f 1e fa          	endbr64
    4cc9:	55                   	push   %rbp
    4cca:	48 89 e5             	mov    %rsp,%rbp
    4ccd:	48 83 ec 10          	sub    $0x10,%rsp
    4cd1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4cd5:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4cd9:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    4cdd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4ce1:	48 89 d6             	mov    %rdx,%rsi
    4ce4:	48 89 c7             	mov    %rax,%rdi
    4ce7:	e8 88 05 00 00       	call   5274 <_ZNSt12_Destroy_auxILb1EE9__destroyIP15BenchmarkResultEEvT_S4_>
    4cec:	90                   	nop
    4ced:	c9                   	leave
    4cee:	c3                   	ret

0000000000004cef <_ZSt7forwardIRK15BenchmarkResultEOT_RNSt16remove_referenceIS3_E4typeE>:
    4cef:	f3 0f 1e fa          	endbr64
    4cf3:	55                   	push   %rbp
    4cf4:	48 89 e5             	mov    %rsp,%rbp
    4cf7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4cfb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4cff:	5d                   	pop    %rbp
    4d00:	c3                   	ret
    4d01:	90                   	nop

0000000000004d02 <_ZNSt15__new_allocatorI15BenchmarkResultE9constructIS0_JRKS0_EEEvPT_DpOT0_>:
    4d02:	f3 0f 1e fa          	endbr64
    4d06:	55                   	push   %rbp
    4d07:	48 89 e5             	mov    %rsp,%rbp
    4d0a:	53                   	push   %rbx
    4d0b:	48 83 ec 28          	sub    $0x28,%rsp
    4d0f:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    4d13:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    4d17:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    4d1b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    4d1f:	48 89 c6             	mov    %rax,%rsi
    4d22:	bf 28 00 00 00       	mov    $0x28,%edi
    4d27:	e8 41 f0 ff ff       	call   3d6d <_ZnwmPv>
    4d2c:	48 89 c3             	mov    %rax,%rbx
    4d2f:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    4d33:	48 89 c7             	mov    %rax,%rdi
    4d36:	e8 b4 ff ff ff       	call   4cef <_ZSt7forwardIRK15BenchmarkResultEOT_RNSt16remove_referenceIS3_E4typeE>
    4d3b:	48 8b 30             	mov    (%rax),%rsi
    4d3e:	48 8b 78 08          	mov    0x8(%rax),%rdi
    4d42:	48 89 33             	mov    %rsi,(%rbx)
    4d45:	48 89 7b 08          	mov    %rdi,0x8(%rbx)
    4d49:	48 8b 70 10          	mov    0x10(%rax),%rsi
    4d4d:	48 8b 78 18          	mov    0x18(%rax),%rdi
    4d51:	48 89 73 10          	mov    %rsi,0x10(%rbx)
    4d55:	48 89 7b 18          	mov    %rdi,0x18(%rbx)
    4d59:	48 8b 40 20          	mov    0x20(%rax),%rax
    4d5d:	48 89 43 20          	mov    %rax,0x20(%rbx)
    4d61:	90                   	nop
    4d62:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    4d66:	c9                   	leave
    4d67:	c3                   	ret

0000000000004d68 <_ZN9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEEC1ERKS2_>:
    4d68:	f3 0f 1e fa          	endbr64
    4d6c:	55                   	push   %rbp
    4d6d:	48 89 e5             	mov    %rsp,%rbp
    4d70:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4d74:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4d78:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    4d7c:	48 8b 10             	mov    (%rax),%rdx
    4d7f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4d83:	48 89 10             	mov    %rdx,(%rax)
    4d86:	90                   	nop
    4d87:	5d                   	pop    %rbp
    4d88:	c3                   	ret
    4d89:	90                   	nop

0000000000004d8a <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE12_M_check_lenEmPKc>:
    4d8a:	f3 0f 1e fa          	endbr64
    4d8e:	55                   	push   %rbp
    4d8f:	48 89 e5             	mov    %rsp,%rbp
    4d92:	53                   	push   %rbx
    4d93:	48 83 ec 48          	sub    $0x48,%rsp
    4d97:	48 89 7d c8          	mov    %rdi,-0x38(%rbp)
    4d9b:	48 89 75 c0          	mov    %rsi,-0x40(%rbp)
    4d9f:	48 89 55 b8          	mov    %rdx,-0x48(%rbp)
    4da3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    4daa:	00 00 
    4dac:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    4db0:	31 c0                	xor    %eax,%eax
    4db2:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    4db6:	48 89 c7             	mov    %rax,%rdi
    4db9:	e8 ca 04 00 00       	call   5288 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE8max_sizeEv>
    4dbe:	48 89 c3             	mov    %rax,%rbx
    4dc1:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    4dc5:	48 89 c7             	mov    %rax,%rdi
    4dc8:	e8 69 f6 ff ff       	call   4436 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE4sizeEv>
    4dcd:	48 29 c3             	sub    %rax,%rbx
    4dd0:	48 89 da             	mov    %rbx,%rdx
    4dd3:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
    4dd7:	48 39 c2             	cmp    %rax,%rdx
    4dda:	0f 92 c0             	setb   %al
    4ddd:	84 c0                	test   %al,%al
    4ddf:	74 0c                	je     4ded <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE12_M_check_lenEmPKc+0x63>
    4de1:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
    4de5:	48 89 c7             	mov    %rax,%rdi
    4de8:	e8 13 d6 ff ff       	call   2400 <_ZSt20__throw_length_errorPKc@plt>
    4ded:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    4df1:	48 89 c7             	mov    %rax,%rdi
    4df4:	e8 3d f6 ff ff       	call   4436 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE4sizeEv>
    4df9:	48 89 c3             	mov    %rax,%rbx
    4dfc:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    4e00:	48 89 c7             	mov    %rax,%rdi
    4e03:	e8 2e f6 ff ff       	call   4436 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE4sizeEv>
    4e08:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    4e0c:	48 8d 55 c0          	lea    -0x40(%rbp),%rdx
    4e10:	48 8d 45 d8          	lea    -0x28(%rbp),%rax
    4e14:	48 89 d6             	mov    %rdx,%rsi
    4e17:	48 89 c7             	mov    %rax,%rdi
    4e1a:	e8 8f 04 00 00       	call   52ae <_ZSt3maxImERKT_S2_S2_>
    4e1f:	48 8b 00             	mov    (%rax),%rax
    4e22:	48 01 d8             	add    %rbx,%rax
    4e25:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    4e29:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    4e2d:	48 89 c7             	mov    %rax,%rdi
    4e30:	e8 01 f6 ff ff       	call   4436 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE4sizeEv>
    4e35:	48 39 45 e0          	cmp    %rax,-0x20(%rbp)
    4e39:	72 12                	jb     4e4d <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE12_M_check_lenEmPKc+0xc3>
    4e3b:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    4e3f:	48 89 c7             	mov    %rax,%rdi
    4e42:	e8 41 04 00 00       	call   5288 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE8max_sizeEv>
    4e47:	48 3b 45 e0          	cmp    -0x20(%rbp),%rax
    4e4b:	73 0e                	jae    4e5b <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE12_M_check_lenEmPKc+0xd1>
    4e4d:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    4e51:	48 89 c7             	mov    %rax,%rdi
    4e54:	e8 2f 04 00 00       	call   5288 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE8max_sizeEv>
    4e59:	eb 04                	jmp    4e5f <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE12_M_check_lenEmPKc+0xd5>
    4e5b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    4e5f:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    4e63:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    4e6a:	00 00 
    4e6c:	74 05                	je     4e73 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE12_M_check_lenEmPKc+0xe9>
    4e6e:	e8 bd d6 ff ff       	call   2530 <__stack_chk_fail@plt>
    4e73:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    4e77:	c9                   	leave
    4e78:	c3                   	ret
    4e79:	90                   	nop

0000000000004e7a <_ZNSt6vectorI15BenchmarkResultSaIS0_EE5beginEv>:
    4e7a:	f3 0f 1e fa          	endbr64
    4e7e:	55                   	push   %rbp
    4e7f:	48 89 e5             	mov    %rsp,%rbp
    4e82:	48 83 ec 20          	sub    $0x20,%rsp
    4e86:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    4e8a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    4e91:	00 00 
    4e93:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    4e97:	31 c0                	xor    %eax,%eax
    4e99:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    4e9d:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    4ea1:	48 89 d6             	mov    %rdx,%rsi
    4ea4:	48 89 c7             	mov    %rax,%rdi
    4ea7:	e8 bc fe ff ff       	call   4d68 <_ZN9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEEC1ERKS2_>
    4eac:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    4eb0:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    4eb4:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    4ebb:	00 00 
    4ebd:	74 05                	je     4ec4 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE5beginEv+0x4a>
    4ebf:	e8 6c d6 ff ff       	call   2530 <__stack_chk_fail@plt>
    4ec4:	c9                   	leave
    4ec5:	c3                   	ret

0000000000004ec6 <_ZN9__gnu_cxxmiIP15BenchmarkResultSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_>:
    4ec6:	f3 0f 1e fa          	endbr64
    4eca:	55                   	push   %rbp
    4ecb:	48 89 e5             	mov    %rsp,%rbp
    4ece:	53                   	push   %rbx
    4ecf:	48 83 ec 18          	sub    $0x18,%rsp
    4ed3:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    4ed7:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    4edb:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4edf:	48 89 c7             	mov    %rax,%rdi
    4ee2:	e8 9f 00 00 00       	call   4f86 <_ZNK9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEE4baseEv>
    4ee7:	48 8b 18             	mov    (%rax),%rbx
    4eea:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    4eee:	48 89 c7             	mov    %rax,%rdi
    4ef1:	e8 90 00 00 00       	call   4f86 <_ZNK9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEE4baseEv>
    4ef6:	48 8b 00             	mov    (%rax),%rax
    4ef9:	48 29 c3             	sub    %rax,%rbx
    4efc:	48 89 da             	mov    %rbx,%rdx
    4eff:	48 c1 fa 03          	sar    $0x3,%rdx
    4f03:	48 b8 cd cc cc cc cc 	movabs $0xcccccccccccccccd,%rax
    4f0a:	cc cc cc 
    4f0d:	48 0f af c2          	imul   %rdx,%rax
    4f11:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    4f15:	c9                   	leave
    4f16:	c3                   	ret
    4f17:	90                   	nop

0000000000004f18 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE11_M_allocateEm>:
    4f18:	f3 0f 1e fa          	endbr64
    4f1c:	55                   	push   %rbp
    4f1d:	48 89 e5             	mov    %rsp,%rbp
    4f20:	48 83 ec 10          	sub    $0x10,%rsp
    4f24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4f28:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4f2c:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    4f31:	74 15                	je     4f48 <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE11_M_allocateEm+0x30>
    4f33:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4f37:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    4f3b:	48 89 d6             	mov    %rdx,%rsi
    4f3e:	48 89 c7             	mov    %rax,%rdi
    4f41:	e8 97 03 00 00       	call   52dd <_ZNSt16allocator_traitsISaI15BenchmarkResultEE8allocateERS1_m>
    4f46:	eb 05                	jmp    4f4d <_ZNSt12_Vector_baseI15BenchmarkResultSaIS0_EE11_M_allocateEm+0x35>
    4f48:	b8 00 00 00 00       	mov    $0x0,%eax
    4f4d:	c9                   	leave
    4f4e:	c3                   	ret

0000000000004f4f <_ZNSt6vectorI15BenchmarkResultSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_>:
    4f4f:	f3 0f 1e fa          	endbr64
    4f53:	55                   	push   %rbp
    4f54:	48 89 e5             	mov    %rsp,%rbp
    4f57:	48 83 ec 20          	sub    $0x20,%rsp
    4f5b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4f5f:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4f63:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    4f67:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
    4f6b:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
    4f6f:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    4f73:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
    4f77:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4f7b:	48 89 c7             	mov    %rax,%rdi
    4f7e:	e8 88 03 00 00       	call   530b <_ZSt12__relocate_aIP15BenchmarkResultS1_SaIS0_EET0_T_S4_S3_RT1_>
    4f83:	c9                   	leave
    4f84:	c3                   	ret
    4f85:	90                   	nop

0000000000004f86 <_ZNK9__gnu_cxx17__normal_iteratorIP15BenchmarkResultSt6vectorIS1_SaIS1_EEE4baseEv>:
    4f86:	f3 0f 1e fa          	endbr64
    4f8a:	55                   	push   %rbp
    4f8b:	48 89 e5             	mov    %rsp,%rbp
    4f8e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4f92:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    4f96:	5d                   	pop    %rbp
    4f97:	c3                   	ret

0000000000004f98 <_ZNSt15__new_allocatorIfE10deallocateEPfm>:
    4f98:	f3 0f 1e fa          	endbr64
    4f9c:	55                   	push   %rbp
    4f9d:	48 89 e5             	mov    %rsp,%rbp
    4fa0:	48 83 ec 20          	sub    $0x20,%rsp
    4fa4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4fa8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4fac:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    4fb0:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    4fb4:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    4fbb:	00 
    4fbc:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    4fc0:	48 89 d6             	mov    %rdx,%rsi
    4fc3:	48 89 c7             	mov    %rax,%rdi
    4fc6:	e8 25 d5 ff ff       	call   24f0 <_ZdlPvm@plt>
    4fcb:	c9                   	leave
    4fcc:	c3                   	ret

0000000000004fcd <_ZNSt12_Destroy_auxILb1EE9__destroyIPfEEvT_S3_>:
    4fcd:	f3 0f 1e fa          	endbr64
    4fd1:	55                   	push   %rbp
    4fd2:	48 89 e5             	mov    %rsp,%rbp
    4fd5:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    4fd9:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    4fdd:	90                   	nop
    4fde:	5d                   	pop    %rbp
    4fdf:	c3                   	ret

0000000000004fe0 <_ZNSt15__new_allocatorIfE9constructIfJfEEEvPT_DpOT0_>:
    4fe0:	f3 0f 1e fa          	endbr64
    4fe4:	55                   	push   %rbp
    4fe5:	48 89 e5             	mov    %rsp,%rbp
    4fe8:	53                   	push   %rbx
    4fe9:	48 83 ec 28          	sub    $0x28,%rsp
    4fed:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    4ff1:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    4ff5:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    4ff9:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    4ffd:	48 89 c6             	mov    %rax,%rsi
    5000:	bf 04 00 00 00       	mov    $0x4,%edi
    5005:	e8 63 ed ff ff       	call   3d6d <_ZnwmPv>
    500a:	48 89 c3             	mov    %rax,%rbx
    500d:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    5011:	48 89 c7             	mov    %rax,%rdi
    5014:	e8 fc f9 ff ff       	call   4a15 <_ZSt7forwardIfEOT_RNSt16remove_referenceIS0_E4typeE>
    5019:	f3 0f 10 00          	movss  (%rax),%xmm0
    501d:	f3 0f 11 03          	movss  %xmm0,(%rbx)
    5021:	90                   	nop
    5022:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    5026:	c9                   	leave
    5027:	c3                   	ret

0000000000005028 <_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc>:
    5028:	f3 0f 1e fa          	endbr64
    502c:	55                   	push   %rbp
    502d:	48 89 e5             	mov    %rsp,%rbp
    5030:	53                   	push   %rbx
    5031:	48 83 ec 48          	sub    $0x48,%rsp
    5035:	48 89 7d c8          	mov    %rdi,-0x38(%rbp)
    5039:	48 89 75 c0          	mov    %rsi,-0x40(%rbp)
    503d:	48 89 55 b8          	mov    %rdx,-0x48(%rbp)
    5041:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    5048:	00 00 
    504a:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    504e:	31 c0                	xor    %eax,%eax
    5050:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    5054:	48 89 c7             	mov    %rax,%rdi
    5057:	e8 16 03 00 00       	call   5372 <_ZNKSt6vectorIfSaIfEE8max_sizeEv>
    505c:	48 89 c3             	mov    %rax,%rbx
    505f:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    5063:	48 89 c7             	mov    %rax,%rdi
    5066:	e8 2d 03 00 00       	call   5398 <_ZNKSt6vectorIfSaIfEE4sizeEv>
    506b:	48 29 c3             	sub    %rax,%rbx
    506e:	48 89 da             	mov    %rbx,%rdx
    5071:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
    5075:	48 39 c2             	cmp    %rax,%rdx
    5078:	0f 92 c0             	setb   %al
    507b:	84 c0                	test   %al,%al
    507d:	74 0c                	je     508b <_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc+0x63>
    507f:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
    5083:	48 89 c7             	mov    %rax,%rdi
    5086:	e8 75 d3 ff ff       	call   2400 <_ZSt20__throw_length_errorPKc@plt>
    508b:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    508f:	48 89 c7             	mov    %rax,%rdi
    5092:	e8 01 03 00 00       	call   5398 <_ZNKSt6vectorIfSaIfEE4sizeEv>
    5097:	48 89 c3             	mov    %rax,%rbx
    509a:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    509e:	48 89 c7             	mov    %rax,%rdi
    50a1:	e8 f2 02 00 00       	call   5398 <_ZNKSt6vectorIfSaIfEE4sizeEv>
    50a6:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    50aa:	48 8d 55 c0          	lea    -0x40(%rbp),%rdx
    50ae:	48 8d 45 d8          	lea    -0x28(%rbp),%rax
    50b2:	48 89 d6             	mov    %rdx,%rsi
    50b5:	48 89 c7             	mov    %rax,%rdi
    50b8:	e8 f1 01 00 00       	call   52ae <_ZSt3maxImERKT_S2_S2_>
    50bd:	48 8b 00             	mov    (%rax),%rax
    50c0:	48 01 d8             	add    %rbx,%rax
    50c3:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    50c7:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    50cb:	48 89 c7             	mov    %rax,%rdi
    50ce:	e8 c5 02 00 00       	call   5398 <_ZNKSt6vectorIfSaIfEE4sizeEv>
    50d3:	48 39 45 e0          	cmp    %rax,-0x20(%rbp)
    50d7:	72 12                	jb     50eb <_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc+0xc3>
    50d9:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    50dd:	48 89 c7             	mov    %rax,%rdi
    50e0:	e8 8d 02 00 00       	call   5372 <_ZNKSt6vectorIfSaIfEE8max_sizeEv>
    50e5:	48 3b 45 e0          	cmp    -0x20(%rbp),%rax
    50e9:	73 0e                	jae    50f9 <_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc+0xd1>
    50eb:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    50ef:	48 89 c7             	mov    %rax,%rdi
    50f2:	e8 7b 02 00 00       	call   5372 <_ZNKSt6vectorIfSaIfEE8max_sizeEv>
    50f7:	eb 04                	jmp    50fd <_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc+0xd5>
    50f9:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    50fd:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    5101:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    5108:	00 00 
    510a:	74 05                	je     5111 <_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc+0xe9>
    510c:	e8 1f d4 ff ff       	call   2530 <__stack_chk_fail@plt>
    5111:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    5115:	c9                   	leave
    5116:	c3                   	ret

0000000000005117 <_ZN9__gnu_cxxmiIPfSt6vectorIfSaIfEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_>:
    5117:	f3 0f 1e fa          	endbr64
    511b:	55                   	push   %rbp
    511c:	48 89 e5             	mov    %rsp,%rbp
    511f:	53                   	push   %rbx
    5120:	48 83 ec 18          	sub    $0x18,%rsp
    5124:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    5128:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    512c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    5130:	48 89 c7             	mov    %rax,%rdi
    5133:	e8 16 f5 ff ff       	call   464e <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv>
    5138:	48 8b 18             	mov    (%rax),%rbx
    513b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    513f:	48 89 c7             	mov    %rax,%rdi
    5142:	e8 07 f5 ff ff       	call   464e <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv>
    5147:	48 8b 00             	mov    (%rax),%rax
    514a:	48 89 da             	mov    %rbx,%rdx
    514d:	48 29 c2             	sub    %rax,%rdx
    5150:	48 89 d0             	mov    %rdx,%rax
    5153:	48 c1 f8 02          	sar    $0x2,%rax
    5157:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    515b:	c9                   	leave
    515c:	c3                   	ret
    515d:	90                   	nop

000000000000515e <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm>:
    515e:	f3 0f 1e fa          	endbr64
    5162:	55                   	push   %rbp
    5163:	48 89 e5             	mov    %rsp,%rbp
    5166:	48 83 ec 10          	sub    $0x10,%rsp
    516a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    516e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    5172:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    5177:	74 15                	je     518e <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm+0x30>
    5179:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    517d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    5181:	48 89 d6             	mov    %rdx,%rsi
    5184:	48 89 c7             	mov    %rax,%rdi
    5187:	e8 33 02 00 00       	call   53bf <_ZNSt16allocator_traitsISaIfEE8allocateERS0_m>
    518c:	eb 05                	jmp    5193 <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm+0x35>
    518e:	b8 00 00 00 00       	mov    $0x0,%eax
    5193:	c9                   	leave
    5194:	c3                   	ret

0000000000005195 <_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_>:
    5195:	f3 0f 1e fa          	endbr64
    5199:	55                   	push   %rbp
    519a:	48 89 e5             	mov    %rsp,%rbp
    519d:	48 83 ec 20          	sub    $0x20,%rsp
    51a1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    51a5:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    51a9:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    51ad:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
    51b1:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
    51b5:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    51b9:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
    51bd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    51c1:	48 89 c7             	mov    %rax,%rdi
    51c4:	e8 24 02 00 00       	call   53ed <_ZSt12__relocate_aIPfS0_SaIfEET0_T_S3_S2_RT1_>
    51c9:	c9                   	leave
    51ca:	c3                   	ret
    51cb:	90                   	nop

00000000000051cc <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEmiEl>:
    51cc:	f3 0f 1e fa          	endbr64
    51d0:	55                   	push   %rbp
    51d1:	48 89 e5             	mov    %rsp,%rbp
    51d4:	48 83 ec 30          	sub    $0x30,%rsp
    51d8:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    51dc:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
    51e0:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    51e7:	00 00 
    51e9:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    51ed:	31 c0                	xor    %eax,%eax
    51ef:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    51f3:	48 8b 00             	mov    (%rax),%rax
    51f6:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
    51fa:	48 c1 e2 02          	shl    $0x2,%rdx
    51fe:	48 f7 da             	neg    %rdx
    5201:	48 01 d0             	add    %rdx,%rax
    5204:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    5208:	48 8d 55 e8          	lea    -0x18(%rbp),%rdx
    520c:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    5210:	48 89 d6             	mov    %rdx,%rsi
    5213:	48 89 c7             	mov    %rax,%rdi
    5216:	e8 11 f4 ff ff       	call   462c <_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC1ERKS1_>
    521b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    521f:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    5223:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    522a:	00 00 
    522c:	74 05                	je     5233 <_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEmiEl+0x67>
    522e:	e8 fd d2 ff ff       	call   2530 <__stack_chk_fail@plt>
    5233:	c9                   	leave
    5234:	c3                   	ret
    5235:	90                   	nop

0000000000005236 <_ZNSt15__new_allocatorI15BenchmarkResultE10deallocateEPS0_m>:
    5236:	f3 0f 1e fa          	endbr64
    523a:	55                   	push   %rbp
    523b:	48 89 e5             	mov    %rsp,%rbp
    523e:	48 83 ec 20          	sub    $0x20,%rsp
    5242:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    5246:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    524a:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    524e:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    5252:	48 89 d0             	mov    %rdx,%rax
    5255:	48 c1 e0 02          	shl    $0x2,%rax
    5259:	48 01 d0             	add    %rdx,%rax
    525c:	48 c1 e0 03          	shl    $0x3,%rax
    5260:	48 89 c2             	mov    %rax,%rdx
    5263:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    5267:	48 89 d6             	mov    %rdx,%rsi
    526a:	48 89 c7             	mov    %rax,%rdi
    526d:	e8 7e d2 ff ff       	call   24f0 <_ZdlPvm@plt>
    5272:	c9                   	leave
    5273:	c3                   	ret

0000000000005274 <_ZNSt12_Destroy_auxILb1EE9__destroyIP15BenchmarkResultEEvT_S4_>:
    5274:	f3 0f 1e fa          	endbr64
    5278:	55                   	push   %rbp
    5279:	48 89 e5             	mov    %rsp,%rbp
    527c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    5280:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    5284:	90                   	nop
    5285:	5d                   	pop    %rbp
    5286:	c3                   	ret
    5287:	90                   	nop

0000000000005288 <_ZNKSt6vectorI15BenchmarkResultSaIS0_EE8max_sizeEv>:
    5288:	f3 0f 1e fa          	endbr64
    528c:	55                   	push   %rbp
    528d:	48 89 e5             	mov    %rsp,%rbp
    5290:	48 83 ec 10          	sub    $0x10,%rsp
    5294:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    5298:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    529c:	48 89 c7             	mov    %rax,%rdi
    529f:	e8 1a 02 00 00       	call   54be <_ZNKSt12_Vector_baseI15BenchmarkResultSaIS0_EE19_M_get_Tp_allocatorEv>
    52a4:	48 89 c7             	mov    %rax,%rdi
    52a7:	e8 a8 01 00 00       	call   5454 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE11_S_max_sizeERKS1_>
    52ac:	c9                   	leave
    52ad:	c3                   	ret

00000000000052ae <_ZSt3maxImERKT_S2_S2_>:
    52ae:	f3 0f 1e fa          	endbr64
    52b2:	55                   	push   %rbp
    52b3:	48 89 e5             	mov    %rsp,%rbp
    52b6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    52ba:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    52be:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    52c2:	48 8b 10             	mov    (%rax),%rdx
    52c5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    52c9:	48 8b 00             	mov    (%rax),%rax
    52cc:	48 39 c2             	cmp    %rax,%rdx
    52cf:	73 06                	jae    52d7 <_ZSt3maxImERKT_S2_S2_+0x29>
    52d1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    52d5:	eb 04                	jmp    52db <_ZSt3maxImERKT_S2_S2_+0x2d>
    52d7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    52db:	5d                   	pop    %rbp
    52dc:	c3                   	ret

00000000000052dd <_ZNSt16allocator_traitsISaI15BenchmarkResultEE8allocateERS1_m>:
    52dd:	f3 0f 1e fa          	endbr64
    52e1:	55                   	push   %rbp
    52e2:	48 89 e5             	mov    %rsp,%rbp
    52e5:	48 83 ec 10          	sub    $0x10,%rsp
    52e9:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    52ed:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    52f1:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    52f5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    52f9:	ba 00 00 00 00       	mov    $0x0,%edx
    52fe:	48 89 ce             	mov    %rcx,%rsi
    5301:	48 89 c7             	mov    %rax,%rdi
    5304:	e8 c7 01 00 00       	call   54d0 <_ZNSt15__new_allocatorI15BenchmarkResultE8allocateEmPKv>
    5309:	c9                   	leave
    530a:	c3                   	ret

000000000000530b <_ZSt12__relocate_aIP15BenchmarkResultS1_SaIS0_EET0_T_S4_S3_RT1_>:
    530b:	f3 0f 1e fa          	endbr64
    530f:	55                   	push   %rbp
    5310:	48 89 e5             	mov    %rsp,%rbp
    5313:	41 54                	push   %r12
    5315:	53                   	push   %rbx
    5316:	48 83 ec 20          	sub    $0x20,%rsp
    531a:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    531e:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    5322:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    5326:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
    532a:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    532e:	48 89 c7             	mov    %rax,%rdi
    5331:	e8 09 02 00 00       	call   553f <_ZSt12__niter_baseIP15BenchmarkResultET_S2_>
    5336:	49 89 c4             	mov    %rax,%r12
    5339:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    533d:	48 89 c7             	mov    %rax,%rdi
    5340:	e8 fa 01 00 00       	call   553f <_ZSt12__niter_baseIP15BenchmarkResultET_S2_>
    5345:	48 89 c3             	mov    %rax,%rbx
    5348:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    534c:	48 89 c7             	mov    %rax,%rdi
    534f:	e8 eb 01 00 00       	call   553f <_ZSt12__niter_baseIP15BenchmarkResultET_S2_>
    5354:	48 89 c7             	mov    %rax,%rdi
    5357:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    535b:	48 89 c1             	mov    %rax,%rcx
    535e:	4c 89 e2             	mov    %r12,%rdx
    5361:	48 89 de             	mov    %rbx,%rsi
    5364:	e8 e8 01 00 00       	call   5551 <_ZSt14__relocate_a_1I15BenchmarkResultS0_ENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS3_E4typeES5_S5_S5_RSaIT0_E>
    5369:	48 83 c4 20          	add    $0x20,%rsp
    536d:	5b                   	pop    %rbx
    536e:	41 5c                	pop    %r12
    5370:	5d                   	pop    %rbp
    5371:	c3                   	ret

0000000000005372 <_ZNKSt6vectorIfSaIfEE8max_sizeEv>:
    5372:	f3 0f 1e fa          	endbr64
    5376:	55                   	push   %rbp
    5377:	48 89 e5             	mov    %rsp,%rbp
    537a:	48 83 ec 10          	sub    $0x10,%rsp
    537e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    5382:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    5386:	48 89 c7             	mov    %rax,%rdi
    5389:	e8 b6 02 00 00       	call   5644 <_ZNKSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
    538e:	48 89 c7             	mov    %rax,%rdi
    5391:	e8 45 02 00 00       	call   55db <_ZNSt6vectorIfSaIfEE11_S_max_sizeERKS0_>
    5396:	c9                   	leave
    5397:	c3                   	ret

0000000000005398 <_ZNKSt6vectorIfSaIfEE4sizeEv>:
    5398:	f3 0f 1e fa          	endbr64
    539c:	55                   	push   %rbp
    539d:	48 89 e5             	mov    %rsp,%rbp
    53a0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    53a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    53a8:	48 8b 50 08          	mov    0x8(%rax),%rdx
    53ac:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    53b0:	48 8b 00             	mov    (%rax),%rax
    53b3:	48 29 c2             	sub    %rax,%rdx
    53b6:	48 89 d0             	mov    %rdx,%rax
    53b9:	48 c1 f8 02          	sar    $0x2,%rax
    53bd:	5d                   	pop    %rbp
    53be:	c3                   	ret

00000000000053bf <_ZNSt16allocator_traitsISaIfEE8allocateERS0_m>:
    53bf:	f3 0f 1e fa          	endbr64
    53c3:	55                   	push   %rbp
    53c4:	48 89 e5             	mov    %rsp,%rbp
    53c7:	48 83 ec 10          	sub    $0x10,%rsp
    53cb:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    53cf:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    53d3:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
    53d7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    53db:	ba 00 00 00 00       	mov    $0x0,%edx
    53e0:	48 89 ce             	mov    %rcx,%rsi
    53e3:	48 89 c7             	mov    %rax,%rdi
    53e6:	e8 6b 02 00 00       	call   5656 <_ZNSt15__new_allocatorIfE8allocateEmPKv>
    53eb:	c9                   	leave
    53ec:	c3                   	ret

00000000000053ed <_ZSt12__relocate_aIPfS0_SaIfEET0_T_S3_S2_RT1_>:
    53ed:	f3 0f 1e fa          	endbr64
    53f1:	55                   	push   %rbp
    53f2:	48 89 e5             	mov    %rsp,%rbp
    53f5:	41 54                	push   %r12
    53f7:	53                   	push   %rbx
    53f8:	48 83 ec 20          	sub    $0x20,%rsp
    53fc:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    5400:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    5404:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    5408:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
    540c:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    5410:	48 89 c7             	mov    %rax,%rdi
    5413:	e8 a3 02 00 00       	call   56bb <_ZSt12__niter_baseIPfET_S1_>
    5418:	49 89 c4             	mov    %rax,%r12
    541b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    541f:	48 89 c7             	mov    %rax,%rdi
    5422:	e8 94 02 00 00       	call   56bb <_ZSt12__niter_baseIPfET_S1_>
    5427:	48 89 c3             	mov    %rax,%rbx
    542a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    542e:	48 89 c7             	mov    %rax,%rdi
    5431:	e8 85 02 00 00       	call   56bb <_ZSt12__niter_baseIPfET_S1_>
    5436:	48 89 c7             	mov    %rax,%rdi
    5439:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    543d:	48 89 c1             	mov    %rax,%rcx
    5440:	4c 89 e2             	mov    %r12,%rdx
    5443:	48 89 de             	mov    %rbx,%rsi
    5446:	e8 82 02 00 00       	call   56cd <_ZSt14__relocate_a_1IffENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS2_E4typeES4_S4_S4_RSaIT0_E>
    544b:	48 83 c4 20          	add    $0x20,%rsp
    544f:	5b                   	pop    %rbx
    5450:	41 5c                	pop    %r12
    5452:	5d                   	pop    %rbp
    5453:	c3                   	ret

0000000000005454 <_ZNSt6vectorI15BenchmarkResultSaIS0_EE11_S_max_sizeERKS1_>:
    5454:	f3 0f 1e fa          	endbr64
    5458:	55                   	push   %rbp
    5459:	48 89 e5             	mov    %rsp,%rbp
    545c:	48 83 ec 30          	sub    $0x30,%rsp
    5460:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    5464:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    546b:	00 00 
    546d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    5471:	31 c0                	xor    %eax,%eax
    5473:	48 b8 33 33 33 33 33 	movabs $0x333333333333333,%rax
    547a:	33 33 03 
    547d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    5481:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    5485:	48 89 c7             	mov    %rax,%rdi
    5488:	e8 a7 02 00 00       	call   5734 <_ZNSt16allocator_traitsISaI15BenchmarkResultEE8max_sizeERKS1_>
    548d:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    5491:	48 8d 55 f0          	lea    -0x10(%rbp),%rdx
    5495:	48 8d 45 e8          	lea    -0x18(%rbp),%rax
    5499:	48 89 d6             	mov    %rdx,%rsi
    549c:	48 89 c7             	mov    %rax,%rdi
    549f:	e8 ae 02 00 00       	call   5752 <_ZSt3minImERKT_S2_S2_>
    54a4:	48 8b 00             	mov    (%rax),%rax
    54a7:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    54ab:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    54b2:	00 00 
    54b4:	74 05                	je     54bb <_ZNSt6vectorI15BenchmarkResultSaIS0_EE11_S_max_sizeERKS1_+0x67>
    54b6:	e8 75 d0 ff ff       	call   2530 <__stack_chk_fail@plt>
    54bb:	c9                   	leave
    54bc:	c3                   	ret
    54bd:	90                   	nop

00000000000054be <_ZNKSt12_Vector_baseI15BenchmarkResultSaIS0_EE19_M_get_Tp_allocatorEv>:
    54be:	f3 0f 1e fa          	endbr64
    54c2:	55                   	push   %rbp
    54c3:	48 89 e5             	mov    %rsp,%rbp
    54c6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    54ca:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    54ce:	5d                   	pop    %rbp
    54cf:	c3                   	ret

00000000000054d0 <_ZNSt15__new_allocatorI15BenchmarkResultE8allocateEmPKv>:
    54d0:	f3 0f 1e fa          	endbr64
    54d4:	55                   	push   %rbp
    54d5:	48 89 e5             	mov    %rsp,%rbp
    54d8:	48 83 ec 20          	sub    $0x20,%rsp
    54dc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    54e0:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    54e4:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    54e8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    54ec:	48 89 c7             	mov    %rax,%rdi
    54ef:	e8 8e 02 00 00       	call   5782 <_ZNKSt15__new_allocatorI15BenchmarkResultE11_M_max_sizeEv>
    54f4:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
    54f8:	0f 92 c0             	setb   %al
    54fb:	0f b6 c0             	movzbl %al,%eax
    54fe:	48 85 c0             	test   %rax,%rax
    5501:	0f 95 c0             	setne  %al
    5504:	84 c0                	test   %al,%al
    5506:	74 1a                	je     5522 <_ZNSt15__new_allocatorI15BenchmarkResultE8allocateEmPKv+0x52>
    5508:	48 b8 66 66 66 66 66 	movabs $0x666666666666666,%rax
    550f:	66 66 06 
    5512:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
    5516:	73 05                	jae    551d <_ZNSt15__new_allocatorI15BenchmarkResultE8allocateEmPKv+0x4d>
    5518:	e8 53 cf ff ff       	call   2470 <_ZSt28__throw_bad_array_new_lengthv@plt>
    551d:	e8 be ce ff ff       	call   23e0 <_ZSt17__throw_bad_allocv@plt>
    5522:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    5526:	48 89 d0             	mov    %rdx,%rax
    5529:	48 c1 e0 02          	shl    $0x2,%rax
    552d:	48 01 d0             	add    %rdx,%rax
    5530:	48 c1 e0 03          	shl    $0x3,%rax
    5534:	48 89 c7             	mov    %rax,%rdi
    5537:	e8 94 cf ff ff       	call   24d0 <_Znwm@plt>
    553c:	90                   	nop
    553d:	c9                   	leave
    553e:	c3                   	ret

000000000000553f <_ZSt12__niter_baseIP15BenchmarkResultET_S2_>:
    553f:	f3 0f 1e fa          	endbr64
    5543:	55                   	push   %rbp
    5544:	48 89 e5             	mov    %rsp,%rbp
    5547:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    554b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    554f:	5d                   	pop    %rbp
    5550:	c3                   	ret

0000000000005551 <_ZSt14__relocate_a_1I15BenchmarkResultS0_ENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS3_E4typeES5_S5_S5_RSaIT0_E>:
    5551:	f3 0f 1e fa          	endbr64
    5555:	55                   	push   %rbp
    5556:	48 89 e5             	mov    %rsp,%rbp
    5559:	48 83 ec 30          	sub    $0x30,%rsp
    555d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    5561:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    5565:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    5569:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
    556d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    5571:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
    5575:	48 c1 f8 03          	sar    $0x3,%rax
    5579:	48 89 c2             	mov    %rax,%rdx
    557c:	48 b8 cd cc cc cc cc 	movabs $0xcccccccccccccccd,%rax
    5583:	cc cc cc 
    5586:	48 0f af c2          	imul   %rdx,%rax
    558a:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    558e:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    5593:	7e 28                	jle    55bd <_ZSt14__relocate_a_1I15BenchmarkResultS0_ENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS3_E4typeES5_S5_S5_RSaIT0_E+0x6c>
    5595:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    5599:	48 89 d0             	mov    %rdx,%rax
    559c:	48 c1 e0 02          	shl    $0x2,%rax
    55a0:	48 01 d0             	add    %rdx,%rax
    55a3:	48 c1 e0 03          	shl    $0x3,%rax
    55a7:	48 89 c2             	mov    %rax,%rdx
    55aa:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
    55ae:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    55b2:	48 89 ce             	mov    %rcx,%rsi
    55b5:	48 89 c7             	mov    %rax,%rdi
    55b8:	e8 23 d0 ff ff       	call   25e0 <memmove@plt>
    55bd:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    55c1:	48 89 d0             	mov    %rdx,%rax
    55c4:	48 c1 e0 02          	shl    $0x2,%rax
    55c8:	48 01 d0             	add    %rdx,%rax
    55cb:	48 c1 e0 03          	shl    $0x3,%rax
    55cf:	48 89 c2             	mov    %rax,%rdx
    55d2:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    55d6:	48 01 d0             	add    %rdx,%rax
    55d9:	c9                   	leave
    55da:	c3                   	ret

00000000000055db <_ZNSt6vectorIfSaIfEE11_S_max_sizeERKS0_>:
    55db:	f3 0f 1e fa          	endbr64
    55df:	55                   	push   %rbp
    55e0:	48 89 e5             	mov    %rsp,%rbp
    55e3:	48 83 ec 30          	sub    $0x30,%rsp
    55e7:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    55eb:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    55f2:	00 00 
    55f4:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    55f8:	31 c0                	xor    %eax,%eax
    55fa:	48 b8 ff ff ff ff ff 	movabs $0x1fffffffffffffff,%rax
    5601:	ff ff 1f 
    5604:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    5608:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    560c:	48 89 c7             	mov    %rax,%rdi
    560f:	e8 86 01 00 00       	call   579a <_ZNSt16allocator_traitsISaIfEE8max_sizeERKS0_>
    5614:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    5618:	48 8d 55 f0          	lea    -0x10(%rbp),%rdx
    561c:	48 8d 45 e8          	lea    -0x18(%rbp),%rax
    5620:	48 89 d6             	mov    %rdx,%rsi
    5623:	48 89 c7             	mov    %rax,%rdi
    5626:	e8 27 01 00 00       	call   5752 <_ZSt3minImERKT_S2_S2_>
    562b:	48 8b 00             	mov    (%rax),%rax
    562e:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    5632:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    5639:	00 00 
    563b:	74 05                	je     5642 <_ZNSt6vectorIfSaIfEE11_S_max_sizeERKS0_+0x67>
    563d:	e8 ee ce ff ff       	call   2530 <__stack_chk_fail@plt>
    5642:	c9                   	leave
    5643:	c3                   	ret

0000000000005644 <_ZNKSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>:
    5644:	f3 0f 1e fa          	endbr64
    5648:	55                   	push   %rbp
    5649:	48 89 e5             	mov    %rsp,%rbp
    564c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    5650:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    5654:	5d                   	pop    %rbp
    5655:	c3                   	ret

0000000000005656 <_ZNSt15__new_allocatorIfE8allocateEmPKv>:
    5656:	f3 0f 1e fa          	endbr64
    565a:	55                   	push   %rbp
    565b:	48 89 e5             	mov    %rsp,%rbp
    565e:	48 83 ec 20          	sub    $0x20,%rsp
    5662:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    5666:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    566a:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    566e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    5672:	48 89 c7             	mov    %rax,%rdi
    5675:	e8 3e 01 00 00       	call   57b8 <_ZNKSt15__new_allocatorIfE11_M_max_sizeEv>
    567a:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
    567e:	0f 92 c0             	setb   %al
    5681:	0f b6 c0             	movzbl %al,%eax
    5684:	48 85 c0             	test   %rax,%rax
    5687:	0f 95 c0             	setne  %al
    568a:	84 c0                	test   %al,%al
    568c:	74 1a                	je     56a8 <_ZNSt15__new_allocatorIfE8allocateEmPKv+0x52>
    568e:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rax
    5695:	ff ff 3f 
    5698:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
    569c:	73 05                	jae    56a3 <_ZNSt15__new_allocatorIfE8allocateEmPKv+0x4d>
    569e:	e8 cd cd ff ff       	call   2470 <_ZSt28__throw_bad_array_new_lengthv@plt>
    56a3:	e8 38 cd ff ff       	call   23e0 <_ZSt17__throw_bad_allocv@plt>
    56a8:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    56ac:	48 c1 e0 02          	shl    $0x2,%rax
    56b0:	48 89 c7             	mov    %rax,%rdi
    56b3:	e8 18 ce ff ff       	call   24d0 <_Znwm@plt>
    56b8:	90                   	nop
    56b9:	c9                   	leave
    56ba:	c3                   	ret

00000000000056bb <_ZSt12__niter_baseIPfET_S1_>:
    56bb:	f3 0f 1e fa          	endbr64
    56bf:	55                   	push   %rbp
    56c0:	48 89 e5             	mov    %rsp,%rbp
    56c3:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    56c7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    56cb:	5d                   	pop    %rbp
    56cc:	c3                   	ret

00000000000056cd <_ZSt14__relocate_a_1IffENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS2_E4typeES4_S4_S4_RSaIT0_E>:
    56cd:	f3 0f 1e fa          	endbr64
    56d1:	55                   	push   %rbp
    56d2:	48 89 e5             	mov    %rsp,%rbp
    56d5:	48 83 ec 30          	sub    $0x30,%rsp
    56d9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    56dd:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    56e1:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    56e5:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
    56e9:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    56ed:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
    56f1:	48 c1 f8 02          	sar    $0x2,%rax
    56f5:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    56f9:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    56fe:	7e 1f                	jle    571f <_ZSt14__relocate_a_1IffENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS2_E4typeES4_S4_S4_RSaIT0_E+0x52>
    5700:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    5704:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    570b:	00 
    570c:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
    5710:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    5714:	48 89 ce             	mov    %rcx,%rsi
    5717:	48 89 c7             	mov    %rax,%rdi
    571a:	e8 c1 ce ff ff       	call   25e0 <memmove@plt>
    571f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    5723:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    572a:	00 
    572b:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    572f:	48 01 d0             	add    %rdx,%rax
    5732:	c9                   	leave
    5733:	c3                   	ret

0000000000005734 <_ZNSt16allocator_traitsISaI15BenchmarkResultEE8max_sizeERKS1_>:
    5734:	f3 0f 1e fa          	endbr64
    5738:	55                   	push   %rbp
    5739:	48 89 e5             	mov    %rsp,%rbp
    573c:	48 83 ec 10          	sub    $0x10,%rsp
    5740:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    5744:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    5748:	48 89 c7             	mov    %rax,%rdi
    574b:	e8 80 00 00 00       	call   57d0 <_ZNKSt15__new_allocatorI15BenchmarkResultE8max_sizeEv>
    5750:	c9                   	leave
    5751:	c3                   	ret

0000000000005752 <_ZSt3minImERKT_S2_S2_>:
    5752:	f3 0f 1e fa          	endbr64
    5756:	55                   	push   %rbp
    5757:	48 89 e5             	mov    %rsp,%rbp
    575a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    575e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    5762:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    5766:	48 8b 10             	mov    (%rax),%rdx
    5769:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    576d:	48 8b 00             	mov    (%rax),%rax
    5770:	48 39 c2             	cmp    %rax,%rdx
    5773:	73 06                	jae    577b <_ZSt3minImERKT_S2_S2_+0x29>
    5775:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    5779:	eb 04                	jmp    577f <_ZSt3minImERKT_S2_S2_+0x2d>
    577b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    577f:	5d                   	pop    %rbp
    5780:	c3                   	ret
    5781:	90                   	nop

0000000000005782 <_ZNKSt15__new_allocatorI15BenchmarkResultE11_M_max_sizeEv>:
    5782:	f3 0f 1e fa          	endbr64
    5786:	55                   	push   %rbp
    5787:	48 89 e5             	mov    %rsp,%rbp
    578a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    578e:	48 b8 33 33 33 33 33 	movabs $0x333333333333333,%rax
    5795:	33 33 03 
    5798:	5d                   	pop    %rbp
    5799:	c3                   	ret

000000000000579a <_ZNSt16allocator_traitsISaIfEE8max_sizeERKS0_>:
    579a:	f3 0f 1e fa          	endbr64
    579e:	55                   	push   %rbp
    579f:	48 89 e5             	mov    %rsp,%rbp
    57a2:	48 83 ec 10          	sub    $0x10,%rsp
    57a6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    57aa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    57ae:	48 89 c7             	mov    %rax,%rdi
    57b1:	e8 38 00 00 00       	call   57ee <_ZNKSt15__new_allocatorIfE8max_sizeEv>
    57b6:	c9                   	leave
    57b7:	c3                   	ret

00000000000057b8 <_ZNKSt15__new_allocatorIfE11_M_max_sizeEv>:
    57b8:	f3 0f 1e fa          	endbr64
    57bc:	55                   	push   %rbp
    57bd:	48 89 e5             	mov    %rsp,%rbp
    57c0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    57c4:	48 b8 ff ff ff ff ff 	movabs $0x1fffffffffffffff,%rax
    57cb:	ff ff 1f 
    57ce:	5d                   	pop    %rbp
    57cf:	c3                   	ret

00000000000057d0 <_ZNKSt15__new_allocatorI15BenchmarkResultE8max_sizeEv>:
    57d0:	f3 0f 1e fa          	endbr64
    57d4:	55                   	push   %rbp
    57d5:	48 89 e5             	mov    %rsp,%rbp
    57d8:	48 83 ec 10          	sub    $0x10,%rsp
    57dc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    57e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    57e4:	48 89 c7             	mov    %rax,%rdi
    57e7:	e8 96 ff ff ff       	call   5782 <_ZNKSt15__new_allocatorI15BenchmarkResultE11_M_max_sizeEv>
    57ec:	c9                   	leave
    57ed:	c3                   	ret

00000000000057ee <_ZNKSt15__new_allocatorIfE8max_sizeEv>:
    57ee:	f3 0f 1e fa          	endbr64
    57f2:	55                   	push   %rbp
    57f3:	48 89 e5             	mov    %rsp,%rbp
    57f6:	48 83 ec 10          	sub    $0x10,%rsp
    57fa:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    57fe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    5802:	48 89 c7             	mov    %rax,%rdi
    5805:	e8 ae ff ff ff       	call   57b8 <_ZNKSt15__new_allocatorIfE11_M_max_sizeEv>
    580a:	c9                   	leave
    580b:	c3                   	ret
    580c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000005810 <atexit>:
    5810:	f3 0f 1e fa          	endbr64
    5814:	48 8b 15 ed 37 00 00 	mov    0x37ed(%rip),%rdx        # 9008 <__dso_handle>
    581b:	31 f6                	xor    %esi,%esi
    581d:	e9 6e cc ff ff       	jmp    2490 <__cxa_atexit@plt>

Disassembly of section .fini:

0000000000005824 <_fini>:
    5824:	f3 0f 1e fa          	endbr64
    5828:	48 83 ec 08          	sub    $0x8,%rsp
    582c:	48 83 c4 08          	add    $0x8,%rsp
    5830:	c3                   	ret
