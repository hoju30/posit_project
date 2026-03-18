; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/dot.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/dot.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double*, double*, i64)* @kernel_dot to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_dot(double* nocapture noundef readonly %0, double* nocapture noundef readonly %1, i64 noundef %2) #0 {
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %27, label %5

5:                                                ; preds = %3
  %6 = and i64 %2, 3
  %7 = icmp ult i64 %2, 4
  br i1 %7, label %10, label %8

8:                                                ; preds = %5
  %9 = and i64 %2, -4
  br label %29

10:                                               ; preds = %29, %5
  %11 = phi double [ undef, %5 ], [ %55, %29 ]
  %12 = phi i64 [ 0, %5 ], [ %56, %29 ]
  %13 = phi double [ 0.000000e+00, %5 ], [ %55, %29 ]
  %14 = icmp eq i64 %6, 0
  br i1 %14, label %27, label %15

15:                                               ; preds = %10, %15
  %16 = phi i64 [ %24, %15 ], [ %12, %10 ]
  %17 = phi double [ %23, %15 ], [ %13, %10 ]
  %18 = phi i64 [ %25, %15 ], [ 0, %10 ]
  %19 = getelementptr inbounds double, double* %0, i64 %16
  %20 = load double, double* %19, align 8, !tbaa !5
  %21 = getelementptr inbounds double, double* %1, i64 %16
  %22 = load double, double* %21, align 8, !tbaa !5
  %23 = tail call double @llvm.fmuladd.f64(double %20, double %22, double %17)
  %24 = add nuw i64 %16, 1
  %25 = add i64 %18, 1
  %26 = icmp eq i64 %25, %6
  br i1 %26, label %27, label %15, !llvm.loop !9

27:                                               ; preds = %10, %15, %3
  %28 = phi double [ 0.000000e+00, %3 ], [ %11, %10 ], [ %23, %15 ]
  ret double %28

29:                                               ; preds = %29, %8
  %30 = phi i64 [ 0, %8 ], [ %56, %29 ]
  %31 = phi double [ 0.000000e+00, %8 ], [ %55, %29 ]
  %32 = phi i64 [ 0, %8 ], [ %57, %29 ]
  %33 = getelementptr inbounds double, double* %0, i64 %30
  %34 = load double, double* %33, align 8, !tbaa !5
  %35 = getelementptr inbounds double, double* %1, i64 %30
  %36 = load double, double* %35, align 8, !tbaa !5
  %37 = tail call double @llvm.fmuladd.f64(double %34, double %36, double %31)
  %38 = or i64 %30, 1
  %39 = getelementptr inbounds double, double* %0, i64 %38
  %40 = load double, double* %39, align 8, !tbaa !5
  %41 = getelementptr inbounds double, double* %1, i64 %38
  %42 = load double, double* %41, align 8, !tbaa !5
  %43 = tail call double @llvm.fmuladd.f64(double %40, double %42, double %37)
  %44 = or i64 %30, 2
  %45 = getelementptr inbounds double, double* %0, i64 %44
  %46 = load double, double* %45, align 8, !tbaa !5
  %47 = getelementptr inbounds double, double* %1, i64 %44
  %48 = load double, double* %47, align 8, !tbaa !5
  %49 = tail call double @llvm.fmuladd.f64(double %46, double %48, double %43)
  %50 = or i64 %30, 3
  %51 = getelementptr inbounds double, double* %0, i64 %50
  %52 = load double, double* %51, align 8, !tbaa !5
  %53 = getelementptr inbounds double, double* %1, i64 %50
  %54 = load double, double* %53, align 8, !tbaa !5
  %55 = tail call double @llvm.fmuladd.f64(double %52, double %54, double %49)
  %56 = add nuw i64 %30, 4
  %57 = add i64 %32, 4
  %58 = icmp eq i64 %57, %9
  br i1 %58, label %10, label %29, !llvm.loop !11
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
