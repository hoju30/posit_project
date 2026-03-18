; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/axpy_sum.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/axpy_sum.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double, double*, double*, i64)* @kernel_axpy_sum to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_axpy_sum(double noundef %0, double* nocapture noundef readonly %1, double* nocapture noundef readonly %2, i64 noundef %3) #0 {
  %5 = icmp eq i64 %3, 0
  br i1 %5, label %29, label %6

6:                                                ; preds = %4
  %7 = and i64 %3, 3
  %8 = icmp ult i64 %3, 4
  br i1 %8, label %11, label %9

9:                                                ; preds = %6
  %10 = and i64 %3, -4
  br label %31

11:                                               ; preds = %31, %6
  %12 = phi double [ undef, %6 ], [ %61, %31 ]
  %13 = phi i64 [ 0, %6 ], [ %62, %31 ]
  %14 = phi double [ 0.000000e+00, %6 ], [ %61, %31 ]
  %15 = icmp eq i64 %7, 0
  br i1 %15, label %29, label %16

16:                                               ; preds = %11, %16
  %17 = phi i64 [ %26, %16 ], [ %13, %11 ]
  %18 = phi double [ %25, %16 ], [ %14, %11 ]
  %19 = phi i64 [ %27, %16 ], [ 0, %11 ]
  %20 = getelementptr inbounds double, double* %1, i64 %17
  %21 = load double, double* %20, align 8, !tbaa !5
  %22 = getelementptr inbounds double, double* %2, i64 %17
  %23 = load double, double* %22, align 8, !tbaa !5
  %24 = tail call double @llvm.fmuladd.f64(double %0, double %21, double %23)
  %25 = fadd double %18, %24
  %26 = add nuw i64 %17, 1
  %27 = add i64 %19, 1
  %28 = icmp eq i64 %27, %7
  br i1 %28, label %29, label %16, !llvm.loop !9

29:                                               ; preds = %11, %16, %4
  %30 = phi double [ 0.000000e+00, %4 ], [ %12, %11 ], [ %25, %16 ]
  ret double %30

31:                                               ; preds = %31, %9
  %32 = phi i64 [ 0, %9 ], [ %62, %31 ]
  %33 = phi double [ 0.000000e+00, %9 ], [ %61, %31 ]
  %34 = phi i64 [ 0, %9 ], [ %63, %31 ]
  %35 = getelementptr inbounds double, double* %1, i64 %32
  %36 = load double, double* %35, align 8, !tbaa !5
  %37 = getelementptr inbounds double, double* %2, i64 %32
  %38 = load double, double* %37, align 8, !tbaa !5
  %39 = tail call double @llvm.fmuladd.f64(double %0, double %36, double %38)
  %40 = fadd double %33, %39
  %41 = or i64 %32, 1
  %42 = getelementptr inbounds double, double* %1, i64 %41
  %43 = load double, double* %42, align 8, !tbaa !5
  %44 = getelementptr inbounds double, double* %2, i64 %41
  %45 = load double, double* %44, align 8, !tbaa !5
  %46 = tail call double @llvm.fmuladd.f64(double %0, double %43, double %45)
  %47 = fadd double %40, %46
  %48 = or i64 %32, 2
  %49 = getelementptr inbounds double, double* %1, i64 %48
  %50 = load double, double* %49, align 8, !tbaa !5
  %51 = getelementptr inbounds double, double* %2, i64 %48
  %52 = load double, double* %51, align 8, !tbaa !5
  %53 = tail call double @llvm.fmuladd.f64(double %0, double %50, double %52)
  %54 = fadd double %47, %53
  %55 = or i64 %32, 3
  %56 = getelementptr inbounds double, double* %1, i64 %55
  %57 = load double, double* %56, align 8, !tbaa !5
  %58 = getelementptr inbounds double, double* %2, i64 %55
  %59 = load double, double* %58, align 8, !tbaa !5
  %60 = tail call double @llvm.fmuladd.f64(double %0, double %57, double %59)
  %61 = fadd double %54, %60
  %62 = add nuw i64 %32, 4
  %63 = add i64 %34, 4
  %64 = icmp eq i64 %63, %10
  br i1 %64, label %11, label %31, !llvm.loop !11
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
