; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/relu_sum.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/relu_sum.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double*, i64)* @kernel_relu_sum to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_relu_sum(double* nocapture noundef readonly %0, i64 noundef %1) #0 {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %20, label %4

4:                                                ; preds = %2
  %5 = and i64 %1, 1
  %6 = icmp eq i64 %1, 1
  br i1 %6, label %9, label %7

7:                                                ; preds = %4
  %8 = and i64 %1, -2
  br label %22

9:                                                ; preds = %22, %4
  %10 = phi double [ undef, %4 ], [ %36, %22 ]
  %11 = phi double [ 0.000000e+00, %4 ], [ %36, %22 ]
  %12 = phi i64 [ 0, %4 ], [ %37, %22 ]
  %13 = icmp eq i64 %5, 0
  br i1 %13, label %20, label %14

14:                                               ; preds = %9
  %15 = getelementptr inbounds double, double* %0, i64 %12
  %16 = load double, double* %15, align 8, !tbaa !5
  %17 = fcmp ogt double %16, 0.000000e+00
  %18 = select i1 %17, double %16, double 0.000000e+00
  %19 = fadd double %11, %18
  br label %20

20:                                               ; preds = %14, %9, %2
  %21 = phi double [ 0.000000e+00, %2 ], [ %10, %9 ], [ %19, %14 ]
  ret double %21

22:                                               ; preds = %22, %7
  %23 = phi double [ 0.000000e+00, %7 ], [ %36, %22 ]
  %24 = phi i64 [ 0, %7 ], [ %37, %22 ]
  %25 = phi i64 [ 0, %7 ], [ %38, %22 ]
  %26 = getelementptr inbounds double, double* %0, i64 %24
  %27 = load double, double* %26, align 8, !tbaa !5
  %28 = fcmp ogt double %27, 0.000000e+00
  %29 = select i1 %28, double %27, double 0.000000e+00
  %30 = fadd double %23, %29
  %31 = or i64 %24, 1
  %32 = getelementptr inbounds double, double* %0, i64 %31
  %33 = load double, double* %32, align 8, !tbaa !5
  %34 = fcmp ogt double %33, 0.000000e+00
  %35 = select i1 %34, double %33, double 0.000000e+00
  %36 = fadd double %30, %35
  %37 = add nuw i64 %24, 2
  %38 = add i64 %25, 2
  %39 = icmp eq i64 %38, %8
  br i1 %39, label %9, label %22, !llvm.loop !9
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
