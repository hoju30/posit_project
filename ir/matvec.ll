; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/matvec.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/matvec.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double*, double*, i64, i64)* @kernel_matvec_sum to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_matvec_sum(double* nocapture noundef readonly %0, double* nocapture noundef readonly %1, i64 noundef %2, i64 noundef %3) #0 {
  %5 = icmp eq i64 %2, 0
  br i1 %5, label %12, label %6

6:                                                ; preds = %4
  %7 = icmp eq i64 %3, 0
  %8 = and i64 %3, 3
  %9 = icmp ult i64 %3, 4
  %10 = and i64 %3, -4
  %11 = icmp eq i64 %8, 0
  br label %14

12:                                               ; preds = %36, %4
  %13 = phi double [ 0.000000e+00, %4 ], [ %38, %36 ]
  ret double %13

14:                                               ; preds = %6, %36
  %15 = phi double [ 0.000000e+00, %6 ], [ %38, %36 ]
  %16 = phi i64 [ 0, %6 ], [ %39, %36 ]
  %17 = mul i64 %16, %3
  br i1 %7, label %36, label %18

18:                                               ; preds = %14
  br i1 %9, label %19, label %41

19:                                               ; preds = %41, %18
  %20 = phi double [ undef, %18 ], [ %71, %41 ]
  %21 = phi i64 [ 0, %18 ], [ %72, %41 ]
  %22 = phi double [ 0.000000e+00, %18 ], [ %71, %41 ]
  br i1 %11, label %36, label %23

23:                                               ; preds = %19, %23
  %24 = phi i64 [ %33, %23 ], [ %21, %19 ]
  %25 = phi double [ %32, %23 ], [ %22, %19 ]
  %26 = phi i64 [ %34, %23 ], [ 0, %19 ]
  %27 = add i64 %24, %17
  %28 = getelementptr inbounds double, double* %0, i64 %27
  %29 = load double, double* %28, align 8, !tbaa !5
  %30 = getelementptr inbounds double, double* %1, i64 %24
  %31 = load double, double* %30, align 8, !tbaa !5
  %32 = tail call double @llvm.fmuladd.f64(double %29, double %31, double %25)
  %33 = add nuw i64 %24, 1
  %34 = add i64 %26, 1
  %35 = icmp eq i64 %34, %8
  br i1 %35, label %36, label %23, !llvm.loop !9

36:                                               ; preds = %19, %23, %14
  %37 = phi double [ 0.000000e+00, %14 ], [ %20, %19 ], [ %32, %23 ]
  %38 = fadd double %15, %37
  %39 = add nuw i64 %16, 1
  %40 = icmp eq i64 %39, %2
  br i1 %40, label %12, label %14, !llvm.loop !11

41:                                               ; preds = %18, %41
  %42 = phi i64 [ %72, %41 ], [ 0, %18 ]
  %43 = phi double [ %71, %41 ], [ 0.000000e+00, %18 ]
  %44 = phi i64 [ %73, %41 ], [ 0, %18 ]
  %45 = add i64 %42, %17
  %46 = getelementptr inbounds double, double* %0, i64 %45
  %47 = load double, double* %46, align 8, !tbaa !5
  %48 = getelementptr inbounds double, double* %1, i64 %42
  %49 = load double, double* %48, align 8, !tbaa !5
  %50 = tail call double @llvm.fmuladd.f64(double %47, double %49, double %43)
  %51 = or i64 %42, 1
  %52 = add i64 %51, %17
  %53 = getelementptr inbounds double, double* %0, i64 %52
  %54 = load double, double* %53, align 8, !tbaa !5
  %55 = getelementptr inbounds double, double* %1, i64 %51
  %56 = load double, double* %55, align 8, !tbaa !5
  %57 = tail call double @llvm.fmuladd.f64(double %54, double %56, double %50)
  %58 = or i64 %42, 2
  %59 = add i64 %58, %17
  %60 = getelementptr inbounds double, double* %0, i64 %59
  %61 = load double, double* %60, align 8, !tbaa !5
  %62 = getelementptr inbounds double, double* %1, i64 %58
  %63 = load double, double* %62, align 8, !tbaa !5
  %64 = tail call double @llvm.fmuladd.f64(double %61, double %63, double %57)
  %65 = or i64 %42, 3
  %66 = add i64 %65, %17
  %67 = getelementptr inbounds double, double* %0, i64 %66
  %68 = load double, double* %67, align 8, !tbaa !5
  %69 = getelementptr inbounds double, double* %1, i64 %65
  %70 = load double, double* %69, align 8, !tbaa !5
  %71 = tail call double @llvm.fmuladd.f64(double %68, double %70, double %64)
  %72 = add nuw i64 %42, 4
  %73 = add i64 %44, 4
  %74 = icmp eq i64 %73, %10
  br i1 %74, label %19, label %41, !llvm.loop !13
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
!13 = distinct !{!13, !12}
