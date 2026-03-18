; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/sum_squares.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/sum_squares.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double*, i64)* @kernel_sum_squares to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_sum_squares(double* nocapture noundef readonly %0, i64 noundef %1) #0 {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %24, label %4

4:                                                ; preds = %2
  %5 = and i64 %1, 3
  %6 = icmp ult i64 %1, 4
  br i1 %6, label %9, label %7

7:                                                ; preds = %4
  %8 = and i64 %1, -4
  br label %26

9:                                                ; preds = %26, %4
  %10 = phi double [ undef, %4 ], [ %44, %26 ]
  %11 = phi double [ 0.000000e+00, %4 ], [ %44, %26 ]
  %12 = phi i64 [ 0, %4 ], [ %45, %26 ]
  %13 = icmp eq i64 %5, 0
  br i1 %13, label %24, label %14

14:                                               ; preds = %9, %14
  %15 = phi double [ %20, %14 ], [ %11, %9 ]
  %16 = phi i64 [ %21, %14 ], [ %12, %9 ]
  %17 = phi i64 [ %22, %14 ], [ 0, %9 ]
  %18 = getelementptr inbounds double, double* %0, i64 %16
  %19 = load double, double* %18, align 8, !tbaa !5
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %19, double %15)
  %21 = add nuw i64 %16, 1
  %22 = add i64 %17, 1
  %23 = icmp eq i64 %22, %5
  br i1 %23, label %24, label %14, !llvm.loop !9

24:                                               ; preds = %9, %14, %2
  %25 = phi double [ 0.000000e+00, %2 ], [ %10, %9 ], [ %20, %14 ]
  ret double %25

26:                                               ; preds = %26, %7
  %27 = phi double [ 0.000000e+00, %7 ], [ %44, %26 ]
  %28 = phi i64 [ 0, %7 ], [ %45, %26 ]
  %29 = phi i64 [ 0, %7 ], [ %46, %26 ]
  %30 = getelementptr inbounds double, double* %0, i64 %28
  %31 = load double, double* %30, align 8, !tbaa !5
  %32 = tail call double @llvm.fmuladd.f64(double %31, double %31, double %27)
  %33 = or i64 %28, 1
  %34 = getelementptr inbounds double, double* %0, i64 %33
  %35 = load double, double* %34, align 8, !tbaa !5
  %36 = tail call double @llvm.fmuladd.f64(double %35, double %35, double %32)
  %37 = or i64 %28, 2
  %38 = getelementptr inbounds double, double* %0, i64 %37
  %39 = load double, double* %38, align 8, !tbaa !5
  %40 = tail call double @llvm.fmuladd.f64(double %39, double %39, double %36)
  %41 = or i64 %28, 3
  %42 = getelementptr inbounds double, double* %0, i64 %41
  %43 = load double, double* %42, align 8, !tbaa !5
  %44 = tail call double @llvm.fmuladd.f64(double %43, double %43, double %40)
  %45 = add nuw i64 %28, 4
  %46 = add i64 %29, 4
  %47 = icmp eq i64 %46, %8
  br i1 %47, label %9, label %26, !llvm.loop !11
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
!10 = !{!"llvm.loop.unroll.disable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
