; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/l1_norm.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/l1_norm.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double*, i64)* @kernel_l1_norm to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_l1_norm(double* nocapture noundef readonly %0, i64 noundef %1) #0 {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %21, label %4

4:                                                ; preds = %2
  %5 = and i64 %1, 1
  %6 = icmp eq i64 %1, 1
  br i1 %6, label %9, label %7

7:                                                ; preds = %4
  %8 = and i64 %1, -2
  br label %23

9:                                                ; preds = %23, %4
  %10 = phi double [ undef, %4 ], [ %39, %23 ]
  %11 = phi double [ 0.000000e+00, %4 ], [ %39, %23 ]
  %12 = phi i64 [ 0, %4 ], [ %40, %23 ]
  %13 = icmp eq i64 %5, 0
  br i1 %13, label %21, label %14

14:                                               ; preds = %9
  %15 = getelementptr inbounds double, double* %0, i64 %12
  %16 = load double, double* %15, align 8, !tbaa !5
  %17 = fcmp oge double %16, 0.000000e+00
  %18 = fneg double %16
  %19 = select i1 %17, double %16, double %18
  %20 = fadd double %11, %19
  br label %21

21:                                               ; preds = %14, %9, %2
  %22 = phi double [ 0.000000e+00, %2 ], [ %10, %9 ], [ %20, %14 ]
  ret double %22

23:                                               ; preds = %23, %7
  %24 = phi double [ 0.000000e+00, %7 ], [ %39, %23 ]
  %25 = phi i64 [ 0, %7 ], [ %40, %23 ]
  %26 = phi i64 [ 0, %7 ], [ %41, %23 ]
  %27 = getelementptr inbounds double, double* %0, i64 %25
  %28 = load double, double* %27, align 8, !tbaa !5
  %29 = fcmp oge double %28, 0.000000e+00
  %30 = fneg double %28
  %31 = select i1 %29, double %28, double %30
  %32 = fadd double %24, %31
  %33 = or i64 %25, 1
  %34 = getelementptr inbounds double, double* %0, i64 %33
  %35 = load double, double* %34, align 8, !tbaa !5
  %36 = fcmp oge double %35, 0.000000e+00
  %37 = fneg double %35
  %38 = select i1 %36, double %35, double %37
  %39 = fadd double %32, %38
  %40 = add nuw i64 %25, 2
  %41 = add i64 %26, 2
  %42 = icmp eq i64 %41, %8
  br i1 %42, label %9, label %23, !llvm.loop !9
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
