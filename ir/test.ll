; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::mersenne_twister_engine" = type { [624 x i64], i64 }
%"class.std::normal_distribution" = type <{ %"struct.std::normal_distribution<>::param_type", double, i8, [7 x i8] }>
%"struct.std::normal_distribution<>::param_type" = type { double, double }

$_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

; Function Attrs: norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %1 = alloca %"class.std::mersenne_twister_engine", align 8
  %2 = alloca %"class.std::normal_distribution", align 16
  %3 = bitcast %"class.std::mersenne_twister_engine"* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 5000, i8* nonnull %3) #8
  %4 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %1, i64 0, i32 0, i64 0
  store i64 42, i64* %4, align 8, !tbaa !5
  br label %5

5:                                                ; preds = %16, %0
  %6 = phi i64 [ 42, %0 ], [ %21, %16 ]
  %7 = phi i64 [ 1, %0 ], [ %23, %16 ]
  %8 = lshr i64 %6, 30
  %9 = xor i64 %8, %6
  %10 = mul nuw nsw i64 %9, 1812433253
  %11 = add nuw i64 %10, %7
  %12 = and i64 %11, 4294967295
  %13 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %1, i64 0, i32 0, i64 %7
  store i64 %12, i64* %13, align 8, !tbaa !5
  %14 = add nuw nsw i64 %7, 1
  %15 = icmp eq i64 %14, 624
  br i1 %15, label %24, label %16, !llvm.loop !9

16:                                               ; preds = %5
  %17 = lshr i64 %12, 30
  %18 = xor i64 %17, %11
  %19 = mul i64 %18, 1812433253
  %20 = add i64 %19, %14
  %21 = and i64 %20, 4294967295
  %22 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %1, i64 0, i32 0, i64 %14
  store i64 %21, i64* %22, align 8, !tbaa !5
  %23 = add nuw nsw i64 %7, 2
  br label %5

24:                                               ; preds = %5
  %25 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %1, i64 0, i32 1
  store i64 624, i64* %25, align 8, !tbaa !11
  %26 = bitcast %"class.std::normal_distribution"* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %26) #8
  %27 = bitcast %"class.std::normal_distribution"* %2 to <2 x double>*
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, <2 x double>* %27, align 16, !tbaa !13
  %28 = getelementptr inbounds %"class.std::normal_distribution", %"class.std::normal_distribution"* %2, i64 0, i32 1
  store double 0.000000e+00, double* %28, align 16, !tbaa !15
  %29 = getelementptr inbounds %"class.std::normal_distribution", %"class.std::normal_distribution"* %2, i64 0, i32 2
  store i8 0, i8* %29, align 8, !tbaa !19
  %30 = getelementptr inbounds %"class.std::normal_distribution", %"class.std::normal_distribution"* %2, i64 0, i32 0
  %31 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %32 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %33 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %34 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %35 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %36 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %37 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %38 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %39 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %40 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %41 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %42 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %43 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %44 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %45 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %46 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %47 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %48 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %49 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  %50 = call noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %2, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %30)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %26) #8
  call void @llvm.lifetime.end.p0i8(i64 5000, i8* nonnull %3) #8
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
declare double @sqrt(double noundef) local_unnamed_addr #2

; Function Attrs: uwtable
define linkonce_odr dso_local noundef double @_ZNSt19normal_distributionIdEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEdRT_RKNS0_10param_typeE(%"class.std::normal_distribution"* noundef nonnull align 8 dereferenceable(25) %0, %"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1, %"struct.std::normal_distribution<>::param_type"* noundef nonnull align 8 dereferenceable(16) %2) local_unnamed_addr #3 comdat align 2 {
  %4 = getelementptr inbounds %"class.std::normal_distribution", %"class.std::normal_distribution"* %0, i64 0, i32 2
  %5 = load i8, i8* %4, align 8, !tbaa !19, !range !20, !noundef !21
  %6 = icmp eq i8 %5, 0
  br i1 %6, label %10, label %7

7:                                                ; preds = %3
  store i8 0, i8* %4, align 8, !tbaa !19
  %8 = getelementptr inbounds %"class.std::normal_distribution", %"class.std::normal_distribution"* %0, i64 0, i32 1
  %9 = load double, double* %8, align 8, !tbaa !15
  br label %78

10:                                               ; preds = %3, %62
  %11 = tail call x86_fp80 @logl(x86_fp80 noundef 0xK401F8000000000000000) #8
  %12 = tail call x86_fp80 @logl(x86_fp80 noundef 0xK40008000000000000000) #8
  %13 = fdiv x86_fp80 %11, %12
  %14 = fptoui x86_fp80 %13 to i64
  %15 = add i64 %14, 52
  %16 = udiv i64 %15, %14
  %17 = tail call i64 @llvm.umax.i64(i64 %16, i64 1)
  br label %21

18:                                               ; preds = %21
  %19 = fdiv double %27, %30
  %20 = fcmp ult double %19, 1.000000e+00
  br i1 %20, label %35, label %33, !prof !22

21:                                               ; preds = %21, %10
  %22 = phi i64 [ %17, %10 ], [ %31, %21 ]
  %23 = phi double [ 1.000000e+00, %10 ], [ %30, %21 ]
  %24 = phi double [ 0.000000e+00, %10 ], [ %27, %21 ]
  %25 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(%"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1)
  %26 = uitofp i64 %25 to double
  %27 = tail call double @llvm.fmuladd.f64(double %26, double %23, double %24)
  %28 = fpext double %23 to x86_fp80
  %29 = fmul x86_fp80 %28, 0xK401F8000000000000000
  %30 = fptrunc x86_fp80 %29 to double
  %31 = add i64 %22, -1
  %32 = icmp eq i64 %31, 0
  br i1 %32, label %18, label %21, !llvm.loop !23

33:                                               ; preds = %18
  %34 = tail call double @nextafter(double noundef 1.000000e+00, double noundef 0.000000e+00) #8
  br label %35

35:                                               ; preds = %18, %33
  %36 = phi double [ %34, %33 ], [ %19, %18 ]
  %37 = tail call x86_fp80 @logl(x86_fp80 noundef 0xK401F8000000000000000) #8
  %38 = tail call x86_fp80 @logl(x86_fp80 noundef 0xK40008000000000000000) #8
  %39 = fdiv x86_fp80 %37, %38
  %40 = fptoui x86_fp80 %39 to i64
  %41 = add i64 %40, 52
  %42 = udiv i64 %41, %40
  %43 = tail call i64 @llvm.umax.i64(i64 %42, i64 1)
  br label %48

44:                                               ; preds = %48
  %45 = tail call double @llvm.fmuladd.f64(double %36, double 2.000000e+00, double -1.000000e+00)
  %46 = fdiv double %54, %57
  %47 = fcmp ult double %46, 1.000000e+00
  br i1 %47, label %62, label %60, !prof !22

48:                                               ; preds = %48, %35
  %49 = phi i64 [ %43, %35 ], [ %58, %48 ]
  %50 = phi double [ 1.000000e+00, %35 ], [ %57, %48 ]
  %51 = phi double [ 0.000000e+00, %35 ], [ %54, %48 ]
  %52 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(%"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %1)
  %53 = uitofp i64 %52 to double
  %54 = tail call double @llvm.fmuladd.f64(double %53, double %50, double %51)
  %55 = fpext double %50 to x86_fp80
  %56 = fmul x86_fp80 %55, 0xK401F8000000000000000
  %57 = fptrunc x86_fp80 %56 to double
  %58 = add i64 %49, -1
  %59 = icmp eq i64 %58, 0
  br i1 %59, label %44, label %48, !llvm.loop !23

60:                                               ; preds = %44
  %61 = tail call double @nextafter(double noundef 1.000000e+00, double noundef 0.000000e+00) #8
  br label %62

62:                                               ; preds = %44, %60
  %63 = phi double [ %61, %60 ], [ %46, %44 ]
  %64 = tail call double @llvm.fmuladd.f64(double %63, double 2.000000e+00, double -1.000000e+00)
  %65 = fmul double %64, %64
  %66 = tail call double @llvm.fmuladd.f64(double %45, double %45, double %65)
  %67 = fcmp ogt double %66, 1.000000e+00
  %68 = fcmp oeq double %66, 0.000000e+00
  %69 = or i1 %67, %68
  br i1 %69, label %10, label %70, !llvm.loop !24

70:                                               ; preds = %62
  %71 = tail call double @log(double noundef %66) #8
  %72 = fmul double %71, -2.000000e+00
  %73 = fdiv double %72, %66
  %74 = tail call double @sqrt(double noundef %73) #8
  %75 = fmul double %45, %74
  %76 = getelementptr inbounds %"class.std::normal_distribution", %"class.std::normal_distribution"* %0, i64 0, i32 1
  store double %75, double* %76, align 8, !tbaa !15
  store i8 1, i8* %4, align 8, !tbaa !19
  %77 = fmul double %64, %74
  br label %78

78:                                               ; preds = %70, %7
  %79 = phi double [ %9, %7 ], [ %77, %70 ]
  %80 = getelementptr inbounds %"struct.std::normal_distribution<>::param_type", %"struct.std::normal_distribution<>::param_type"* %2, i64 0, i32 1
  %81 = load double, double* %80, align 8, !tbaa !25
  %82 = getelementptr inbounds %"struct.std::normal_distribution<>::param_type", %"struct.std::normal_distribution<>::param_type"* %2, i64 0, i32 0
  %83 = load double, double* %82, align 8, !tbaa !26
  %84 = tail call double @llvm.fmuladd.f64(double %79, double %81, double %83)
  ret double %84
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #4

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
declare double @log(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(%"class.std::mersenne_twister_engine"* noundef nonnull align 8 dereferenceable(5000) %0) local_unnamed_addr #5 comdat align 2 {
  %2 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 1
  %3 = load i64, i64* %2, align 8, !tbaa !11
  %4 = icmp ugt i64 %3, 623
  br i1 %4, label %5, label %93

5:                                                ; preds = %1
  %6 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 0
  %7 = load i64, i64* %6, align 8, !tbaa !5
  %8 = insertelement <2 x i64> poison, i64 %7, i64 1
  br label %9

9:                                                ; preds = %9, %5
  %10 = phi i64 [ 0, %5 ], [ %32, %9 ]
  %11 = phi <2 x i64> [ %8, %5 ], [ %16, %9 ]
  %12 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %10
  %13 = or i64 %10, 1
  %14 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %13
  %15 = bitcast i64* %14 to <2 x i64>*
  %16 = load <2 x i64>, <2 x i64>* %15, align 8, !tbaa !5
  %17 = shufflevector <2 x i64> %11, <2 x i64> %16, <2 x i32> <i32 1, i32 2>
  %18 = and <2 x i64> %17, <i64 -2147483648, i64 -2147483648>
  %19 = and <2 x i64> %16, <i64 2147483646, i64 2147483646>
  %20 = or <2 x i64> %19, %18
  %21 = add nuw nsw i64 %10, 397
  %22 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %21
  %23 = bitcast i64* %22 to <2 x i64>*
  %24 = load <2 x i64>, <2 x i64>* %23, align 8, !tbaa !5
  %25 = lshr exact <2 x i64> %20, <i64 1, i64 1>
  %26 = xor <2 x i64> %25, %24
  %27 = and <2 x i64> %16, <i64 1, i64 1>
  %28 = icmp eq <2 x i64> %27, zeroinitializer
  %29 = select <2 x i1> %28, <2 x i64> zeroinitializer, <2 x i64> <i64 2567483615, i64 2567483615>
  %30 = xor <2 x i64> %26, %29
  %31 = bitcast i64* %12 to <2 x i64>*
  store <2 x i64> %30, <2 x i64>* %31, align 8, !tbaa !5
  %32 = add nuw i64 %10, 2
  %33 = icmp eq i64 %32, 226
  br i1 %33, label %34, label %9, !llvm.loop !27

34:                                               ; preds = %9
  %35 = extractelement <2 x i64> %16, i64 1
  %36 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 226
  %37 = and i64 %35, -2147483648
  %38 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 227
  %39 = load i64, i64* %38, align 8, !tbaa !5
  %40 = and i64 %39, 2147483646
  %41 = or i64 %40, %37
  %42 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 623
  %43 = load i64, i64* %42, align 8, !tbaa !5
  %44 = lshr exact i64 %41, 1
  %45 = xor i64 %44, %43
  %46 = and i64 %39, 1
  %47 = icmp eq i64 %46, 0
  %48 = select i1 %47, i64 0, i64 2567483615
  %49 = xor i64 %45, %48
  store i64 %49, i64* %36, align 8, !tbaa !5
  %50 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 227
  %51 = load i64, i64* %50, align 8, !tbaa !5
  %52 = insertelement <2 x i64> poison, i64 %51, i64 1
  br label %53

53:                                               ; preds = %53, %34
  %54 = phi i64 [ 0, %34 ], [ %76, %53 ]
  %55 = phi <2 x i64> [ %52, %34 ], [ %61, %53 ]
  %56 = add i64 %54, 227
  %57 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %56
  %58 = add i64 %54, 228
  %59 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %58
  %60 = bitcast i64* %59 to <2 x i64>*
  %61 = load <2 x i64>, <2 x i64>* %60, align 8, !tbaa !5
  %62 = shufflevector <2 x i64> %55, <2 x i64> %61, <2 x i32> <i32 1, i32 2>
  %63 = and <2 x i64> %62, <i64 -2147483648, i64 -2147483648>
  %64 = and <2 x i64> %61, <i64 2147483646, i64 2147483646>
  %65 = or <2 x i64> %64, %63
  %66 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %54
  %67 = bitcast i64* %66 to <2 x i64>*
  %68 = load <2 x i64>, <2 x i64>* %67, align 8, !tbaa !5
  %69 = lshr exact <2 x i64> %65, <i64 1, i64 1>
  %70 = xor <2 x i64> %69, %68
  %71 = and <2 x i64> %61, <i64 1, i64 1>
  %72 = icmp eq <2 x i64> %71, zeroinitializer
  %73 = select <2 x i1> %72, <2 x i64> zeroinitializer, <2 x i64> <i64 2567483615, i64 2567483615>
  %74 = xor <2 x i64> %70, %73
  %75 = bitcast i64* %57 to <2 x i64>*
  store <2 x i64> %74, <2 x i64>* %75, align 8, !tbaa !5
  %76 = add nuw i64 %54, 2
  %77 = icmp eq i64 %76, 396
  br i1 %77, label %78, label %53, !llvm.loop !30

78:                                               ; preds = %53
  %79 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 623
  %80 = load i64, i64* %79, align 8, !tbaa !5
  %81 = and i64 %80, -2147483648
  %82 = load i64, i64* %6, align 8, !tbaa !5
  %83 = and i64 %82, 2147483646
  %84 = or i64 %83, %81
  %85 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 396
  %86 = load i64, i64* %85, align 8, !tbaa !5
  %87 = lshr exact i64 %84, 1
  %88 = xor i64 %87, %86
  %89 = and i64 %82, 1
  %90 = icmp eq i64 %89, 0
  %91 = select i1 %90, i64 0, i64 2567483615
  %92 = xor i64 %88, %91
  store i64 %92, i64* %79, align 8, !tbaa !5
  br label %93

93:                                               ; preds = %78, %1
  %94 = phi i64 [ 0, %78 ], [ %3, %1 ]
  %95 = add nuw nsw i64 %94, 1
  store i64 %95, i64* %2, align 8, !tbaa !11
  %96 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %94
  %97 = load i64, i64* %96, align 8, !tbaa !5
  %98 = lshr i64 %97, 11
  %99 = and i64 %98, 4294967295
  %100 = xor i64 %99, %97
  %101 = shl i64 %100, 7
  %102 = and i64 %101, 2636928640
  %103 = xor i64 %102, %100
  %104 = shl i64 %103, 15
  %105 = and i64 %104, 4022730752
  %106 = xor i64 %105, %103
  %107 = lshr i64 %106, 18
  %108 = xor i64 %107, %106
  ret i64 %108
}

; Function Attrs: nounwind
declare double @nextafter(double noundef, double noundef) local_unnamed_addr #6

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
declare x86_fp80 @logl(x86_fp80 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #7

attributes #0 = { norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(write) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nounwind }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 16.0.6 (23ubuntu4)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"long", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!12, !6, i64 4992}
!12 = !{!"_ZTSSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE", !7, i64 0, !6, i64 4992}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !7, i64 0}
!15 = !{!16, !14, i64 16}
!16 = !{!"_ZTSSt19normal_distributionIdE", !17, i64 0, !14, i64 16, !18, i64 24}
!17 = !{!"_ZTSNSt19normal_distributionIdE10param_typeE", !14, i64 0, !14, i64 8}
!18 = !{!"bool", !7, i64 0}
!19 = !{!16, !18, i64 24}
!20 = !{i8 0, i8 2}
!21 = !{}
!22 = !{!"branch_weights", i32 2000, i32 1}
!23 = distinct !{!23, !10}
!24 = distinct !{!24, !10}
!25 = !{!17, !14, i64 8}
!26 = !{!17, !14, i64 0}
!27 = distinct !{!27, !10, !28, !29}
!28 = !{!"llvm.loop.isvectorized", i32 1}
!29 = !{!"llvm.loop.unroll.runtime.disable"}
!30 = distinct !{!30, !10, !28, !29}
