; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/axpby_sum.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/axpby_sum.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double, double*, double, double*, i64)* @kernel_axpby_sum to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_axpby_sum(double noundef %0, double* nocapture noundef readonly %1, double noundef %2, double* nocapture noundef readonly %3, i64 noundef %4) #0 {
  %6 = icmp eq i64 %4, 0
  br i1 %6, label %25, label %7

7:                                                ; preds = %5
  %8 = and i64 %4, 1
  %9 = icmp eq i64 %4, 1
  br i1 %9, label %12, label %10

10:                                               ; preds = %7
  %11 = and i64 %4, -2
  br label %27

12:                                               ; preds = %27, %7
  %13 = phi double [ undef, %7 ], [ %45, %27 ]
  %14 = phi i64 [ 0, %7 ], [ %46, %27 ]
  %15 = phi double [ 0.000000e+00, %7 ], [ %45, %27 ]
  %16 = icmp eq i64 %8, 0
  br i1 %16, label %25, label %17

17:                                               ; preds = %12
  %18 = getelementptr inbounds double, double* %1, i64 %14
  %19 = load double, double* %18, align 8, !tbaa !5
  %20 = getelementptr inbounds double, double* %3, i64 %14
  %21 = load double, double* %20, align 8, !tbaa !5
  %22 = fmul double %21, %2
  %23 = tail call double @llvm.fmuladd.f64(double %0, double %19, double %22)
  %24 = fadd double %15, %23
  br label %25

25:                                               ; preds = %17, %12, %5
  %26 = phi double [ 0.000000e+00, %5 ], [ %13, %12 ], [ %24, %17 ]
  ret double %26

27:                                               ; preds = %27, %10
  %28 = phi i64 [ 0, %10 ], [ %46, %27 ]
  %29 = phi double [ 0.000000e+00, %10 ], [ %45, %27 ]
  %30 = phi i64 [ 0, %10 ], [ %47, %27 ]
  %31 = getelementptr inbounds double, double* %1, i64 %28
  %32 = load double, double* %31, align 8, !tbaa !5
  %33 = getelementptr inbounds double, double* %3, i64 %28
  %34 = load double, double* %33, align 8, !tbaa !5
  %35 = fmul double %34, %2
  %36 = tail call double @llvm.fmuladd.f64(double %0, double %32, double %35)
  %37 = fadd double %29, %36
  %38 = or i64 %28, 1
  %39 = getelementptr inbounds double, double* %1, i64 %38
  %40 = load double, double* %39, align 8, !tbaa !5
  %41 = getelementptr inbounds double, double* %3, i64 %38
  %42 = load double, double* %41, align 8, !tbaa !5
  %43 = fmul double %42, %2
  %44 = tail call double @llvm.fmuladd.f64(double %0, double %40, double %43)
  %45 = fadd double %37, %44
  %46 = add nuw i64 %28, 2
  %47 = add i64 %30, 2
  %48 = icmp eq i64 %47, %11
  br i1 %48, label %12, label %27, !llvm.loop !9
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { mustprogress nofree noinline nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 16.0.6 (23ubuntu4)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
